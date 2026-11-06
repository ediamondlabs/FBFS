"""Natural-user FBFS pilot on the ExtraSensory activity dataset."""

import argparse
import copy
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "federated"))
from common import (  # noqa: E402
    bfs_federated,
    bfs_federated_diverse,
    data_iter,
    evaluate,
    evaluate_ensemble,
    fedavg_gate_weights,
    make_difflogic_net,
    set_gate_weights,
    train_one_client,
)


LABEL_COLUMNS = [
    "label:LYING_DOWN",
    "label:SITTING",
    "label:OR_standing",
    "label:FIX_walking",
    "label:FIX_running",
    "label:BICYCLING",
]
LABEL_NAMES = [
    "lying",
    "sitting",
    "standing",
    "walking",
    "running",
    "bicycling",
]


def _read_uuid_list(fold_dir, fold, split):
    pattern = str(Path(fold_dir) / f"fold_{fold}_{split}_*_uuids.txt")
    paths = sorted(glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No fold files match {pattern}")
    return {
        line.strip()
        for path in paths
        for line in Path(path).read_text().splitlines()
        if line.strip()
    }


def _load_users(feature_dir, feature_set):
    paths = sorted(Path(feature_dir).glob("*.features_labels.csv.gz"))
    if not paths:
        raise FileNotFoundError(f"No ExtraSensory user files in {feature_dir}")

    header = pd.read_csv(paths[0], nrows=0).columns.tolist()
    if feature_set == "accelerometer":
        feature_columns = [
            name for name in header if name.startswith("raw_acc:")
        ]
    else:
        feature_columns = [
            name
            for name in header
            if name != "timestamp"
            and name != "label_source"
            and not name.startswith("label:")
        ]
    usecols = feature_columns + LABEL_COLUMNS
    users = {}
    for path in paths:
        user_id = path.name.split(".")[0]
        frame = pd.read_csv(path, usecols=usecols)
        labels = frame[LABEL_COLUMNS].to_numpy(dtype=np.float32)
        positive = np.nan_to_num(labels, nan=0.0) == 1.0
        valid = positive.sum(axis=1) == 1
        users[user_id] = {
            "x": frame.loc[valid, feature_columns].to_numpy(dtype=np.float32),
            "y": positive[valid].argmax(axis=1).astype(np.int64),
        }
    return users, feature_columns


def _shared_thresholds(users, train_ids):
    per_user = []
    for user_id in sorted(train_ids):
        x = users[user_id]["x"]
        with np.errstate(all="ignore"):
            per_user.append(np.nanquantile(x, [0.25, 0.5, 0.75], axis=0).T)
    stacked = np.stack(per_user)
    with np.errstate(all="ignore"):
        thresholds = np.nanmean(stacked, axis=0)
    thresholds = np.nan_to_num(thresholds, nan=0.0, posinf=0.0, neginf=0.0)
    return thresholds.astype(np.float32)


def _encode(x, thresholds):
    finite = np.isfinite(x)
    filled = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    bits = [
        finite & (filled > thresholds[:, threshold_id])
        for threshold_id in range(thresholds.shape[1])
    ]
    bits.append(finite)
    return np.concatenate(bits, axis=1).astype(np.float32)


def _sample_client(y, fraction, minimum, maximum, rng):
    count = min(
        len(y),
        maximum,
        max(minimum, int(round(fraction * len(y)))),
    )
    indices = np.arange(len(y))
    rng.shuffle(indices)
    return indices[:count]


def _random_split(indices, holdout_fraction, rng):
    indices = np.asarray(indices).copy()
    rng.shuffle(indices)
    holdout_count = max(1, int(round(holdout_fraction * len(indices))))
    holdout_count = min(holdout_count, len(indices) - 1)
    return indices[holdout_count:], indices[:holdout_count]


def _balanced_test(users, test_ids, thresholds, per_class, seed):
    x = np.concatenate([users[user_id]["x"] for user_id in sorted(test_ids)])
    y = np.concatenate([users[user_id]["y"] for user_id in sorted(test_ids)])
    x = _encode(x, thresholds)
    rng = np.random.default_rng(seed)
    selected = []
    original_counts = []
    for class_id in range(len(LABEL_COLUMNS)):
        class_indices = np.flatnonzero(y == class_id)
        if not len(class_indices):
            raise ValueError(f"No held-out examples for {LABEL_NAMES[class_id]}")
        rng.shuffle(class_indices)
        original_counts.append(int(len(class_indices)))
        if len(class_indices) >= per_class:
            chosen = class_indices[:per_class]
        else:
            repeats = int(np.ceil(per_class / len(class_indices)))
            chosen = np.tile(class_indices, repeats)[:per_class]
        selected.extend(chosen.tolist())
    rng.shuffle(selected)
    return (
        torch.utils.data.TensorDataset(
            torch.from_numpy(x[selected]),
            torch.from_numpy(y[selected]),
        ),
        original_counts,
    )


def _loader(dataset, batch_size=128, shuffle=False):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=min(batch_size, max(1, len(dataset))),
        shuffle=shuffle,
        drop_last=False,
    )


def _weighted_gate_weights(models, sizes):
    weights = torch.tensor(sizes, dtype=torch.float32)
    weights /= weights.sum()
    result = {}
    for name, _ in models[0].named_parameters():
        stacked = torch.stack(
            [dict(model.named_parameters())[name].data for model in models]
        )
        view = (len(models),) + (1,) * (stacked.ndim - 1)
        result[name] = (stacked * weights.to(stacked.device).view(view)).sum(0)
    return result


def _train_class_weighted(model, loader, num_iters, class_weights, lr, device):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    batches = data_iter(loader)
    weights = class_weights.to(device)
    model.train()
    for _ in range(num_iters):
        x, y = next(batches)
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x.view(x.size(0), -1)), y, weight=weights)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    return model


def _js_divergence(distribution, pooled):
    distribution = distribution.astype(np.float64)
    pooled = pooled.astype(np.float64)
    distribution /= distribution.sum()
    pooled /= pooled.sum()
    midpoint = 0.5 * (distribution + pooled)

    def kl(left, right):
        mask = left > 0
        return float(np.sum(left[mask] * np.log2(left[mask] / right[mask])))

    return 0.5 * kl(distribution, midpoint) + 0.5 * kl(pooled, midpoint)


def _partition_audit(users, train_ids):
    counts = np.stack(
        [
            np.bincount(users[user_id]["y"], minlength=len(LABEL_COLUMNS))
            for user_id in sorted(train_ids)
        ]
    )
    sizes = counts.sum(axis=1)
    pooled = counts.sum(axis=0)
    divergences = [_js_divergence(row, pooled) for row in counts]
    coverage = (counts > 0).sum(axis=1)
    return {
        "raw_client_size_min": int(sizes.min()),
        "raw_client_size_median": float(np.median(sizes)),
        "raw_client_size_max": int(sizes.max()),
        "pooled_class_counts": pooled.tolist(),
        "users_per_class": (counts > 0).sum(axis=0).tolist(),
        "class_coverage_histogram": {
            str(value): int((coverage == value).sum())
            for value in range(1, len(LABEL_COLUMNS) + 1)
        },
        "client_to_pooled_js_bits": {
            "mean": float(np.mean(divergences)),
            "median": float(np.median(divergences)),
            "max": float(np.max(divergences)),
        },
    }


def _summary(raw):
    return {
        method: {
            "mean": float(100 * np.mean(values)),
            "std": (
                float(100 * np.std(values, ddof=1)) if len(values) > 1 else 0.0
            ),
            "n": len(values),
        }
        for method, values in raw.items()
        if values
    }


def _save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--feature-dir", default="data-extrasensory/features"
    )
    parser.add_argument(
        "--fold-dir", default="data-extrasensory/folds/cv_5_folds"
    )
    parser.add_argument(
        "--feature-set",
        choices=["accelerometer", "all"],
        default="accelerometer",
    )
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--sample-fraction", type=float, default=0.2)
    parser.add_argument("--min-examples-per-client", type=int, default=100)
    parser.add_argument("--max-examples-per-client", type=int, default=1500)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--test-per-class", type=int, default=1000)
    parser.add_argument("--test-sampling-seed", type=int, default=20260727)
    parser.add_argument("--neurons", type=int, default=300)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--num-iters", type=int, default=500)
    parser.add_argument("--central-iters", type=int, default=3000)
    parser.add_argument("--num-select", type=int, default=100)
    parser.add_argument("--shortlist-per-class", type=int, default=200)
    parser.add_argument("--diversity-lambda", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--out",
        default="scripts/results/rebuttal_natural_extrasensory.json",
    )
    args = parser.parse_args()

    if args.neurons % len(LABEL_COLUMNS):
        raise ValueError("--neurons must be divisible by six")

    users, feature_columns = _load_users(args.feature_dir, args.feature_set)
    train_ids = _read_uuid_list(args.fold_dir, args.fold, "train")
    test_ids = _read_uuid_list(args.fold_dir, args.fold, "test")
    missing = (train_ids | test_ids) - set(users)
    if missing:
        raise ValueError(f"Fold references missing users: {sorted(missing)}")

    thresholds = _shared_thresholds(users, train_ids)
    test_dataset, test_original_counts = _balanced_test(
        users,
        test_ids,
        thresholds,
        args.test_per_class,
        args.test_sampling_seed,
    )
    test_loader = _loader(test_dataset, batch_size=512)
    audit = _partition_audit(users, train_ids)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    methods = [
        "centralized",
        "centralized_class_weighted",
        "client_mean",
        "fedavg_uniform",
        "fedavg_weighted",
        "fedavg_uniform_class_weighted",
        "fedavg_weighted_class_weighted",
        "ensemble",
        "fbfs",
        "diverse_fbfs",
    ]
    raw = {method: [] for method in methods}
    seed_details = {}
    output_path = Path(args.out)
    total_start = time.time()
    print(
        f"ExtraSensory fold={args.fold}, device={device}, "
        f"train_clients={len(train_ids)}, test_clients={len(test_ids)}, "
        f"input_dim={len(feature_columns) * 4}",
        flush=True,
    )

    for seed in args.seeds:
        seed_start = time.time()
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        train_datasets = []
        selection_datasets = []
        client_details = []
        for user_id in sorted(train_ids):
            x = _encode(users[user_id]["x"], thresholds)
            y = users[user_id]["y"]
            sampled = _sample_client(
                y,
                args.sample_fraction,
                args.min_examples_per_client,
                args.max_examples_per_client,
                rng,
            )
            train_indices, selection_indices = _random_split(
                sampled, args.holdout_frac, rng
            )
            train_datasets.append(
                torch.utils.data.TensorDataset(
                    torch.from_numpy(x[train_indices]),
                    torch.from_numpy(y[train_indices]),
                )
            )
            selection_datasets.append(
                torch.utils.data.TensorDataset(
                    torch.from_numpy(x[selection_indices]),
                    torch.from_numpy(y[selection_indices]),
                )
            )
            sampled_counts = np.bincount(
                y[sampled], minlength=len(LABEL_COLUMNS)
            )
            client_details.append(
                {
                    "user_id": user_id,
                    "raw_examples": int(len(y)),
                    "sampled_examples": int(len(sampled)),
                    "train_examples": int(len(train_indices)),
                    "selection_examples": int(len(selection_indices)),
                    "sampled_class_counts": sampled_counts.tolist(),
                }
            )

        train_loaders = [
            _loader(dataset, shuffle=True) for dataset in train_datasets
        ]
        selection_loaders = [
            _loader(dataset) for dataset in selection_datasets
        ]
        client_sizes = [len(dataset) for dataset in train_datasets]
        base_model = make_difflogic_net(
            len(feature_columns) * 4,
            args.neurons,
            args.layers,
            len(LABEL_COLUMNS),
            tau=10.0,
        )

        client_models = []
        for client_id, loader in enumerate(train_loaders):
            model = copy.deepcopy(base_model).to(device)
            train_one_client(
                model,
                loader,
                args.num_iters,
                lr=args.lr,
                device=device,
                binarize=False,
                flat=True,
            )
            client_models.append(model)
            if (client_id + 1) % 8 == 0:
                print(
                    f"  seed={seed} clients trained "
                    f"{client_id + 1}/{len(train_loaders)}",
                    flush=True,
                )

        client_accuracies = [
            evaluate(
                model,
                test_loader,
                device,
                binarize=False,
                hard=True,
                flat=True,
            )
            for model in client_models
        ]
        raw["client_mean"].append(float(np.mean(client_accuracies)))
        raw["ensemble"].append(
            float(
                evaluate_ensemble(
                    client_models,
                    test_loader,
                    device,
                    binarize=False,
                    hard=True,
                    flat=True,
                )
            )
        )

        fedavg_model = copy.deepcopy(base_model).to(device)
        set_gate_weights(fedavg_model, fedavg_gate_weights(client_models))
        raw["fedavg_uniform"].append(
            float(
                evaluate(
                    fedavg_model,
                    test_loader,
                    device,
                    binarize=False,
                    hard=True,
                    flat=True,
                )
            )
        )
        weighted_model = copy.deepcopy(base_model).to(device)
        set_gate_weights(
            weighted_model,
            _weighted_gate_weights(client_models, client_sizes),
        )
        raw["fedavg_weighted"].append(
            float(
                evaluate(
                    weighted_model,
                    test_loader,
                    device,
                    binarize=False,
                    hard=True,
                    flat=True,
                )
            )
        )

        fbfs_accuracy, _ = bfs_federated(
            client_models,
            selection_loaders,
            test_loader,
            device,
            num_classes=len(LABEL_COLUMNS),
            num_select=args.num_select,
            binarize=False,
            flat=True,
        )
        raw["fbfs"].append(float(fbfs_accuracy))
        diverse_accuracy, _, diverse_comm = bfs_federated_diverse(
            client_models,
            selection_loaders,
            test_loader,
            device,
            num_classes=len(LABEL_COLUMNS),
            num_select=args.num_select,
            binarize=False,
            flat=True,
            diversity_lambda=args.diversity_lambda,
            shortlist_per_class=args.shortlist_per_class,
            return_comm=True,
        )
        raw["diverse_fbfs"].append(float(diverse_accuracy))

        central_model = copy.deepcopy(base_model).to(device)
        train_one_client(
            central_model,
            _loader(
                torch.utils.data.ConcatDataset(train_datasets),
                shuffle=True,
            ),
            args.central_iters,
            lr=args.lr,
            device=device,
            binarize=False,
            flat=True,
        )
        raw["centralized"].append(
            float(
                evaluate(
                    central_model,
                    test_loader,
                    device,
                    binarize=False,
                    hard=True,
                    flat=True,
                )
            )
        )

        pooled_train_counts = torch.zeros(len(LABEL_COLUMNS))
        for dataset in train_datasets:
            pooled_train_counts += torch.bincount(
                dataset.tensors[1], minlength=len(LABEL_COLUMNS)
            )
        class_weights = pooled_train_counts.sum() / (
            len(LABEL_COLUMNS) * pooled_train_counts.clamp(min=1)
        )
        weighted_client_models = []
        for client_id, loader in enumerate(train_loaders):
            model = copy.deepcopy(base_model).to(device)
            _train_class_weighted(
                model,
                loader,
                args.num_iters,
                class_weights,
                args.lr,
                device,
            )
            weighted_client_models.append(model)
            if (client_id + 1) % 16 == 0:
                print(
                    f"  seed={seed} weighted controls trained "
                    f"{client_id + 1}/{len(train_loaders)}",
                    flush=True,
                )

        weighted_uniform_model = copy.deepcopy(base_model).to(device)
        set_gate_weights(
            weighted_uniform_model,
            fedavg_gate_weights(weighted_client_models),
        )
        raw["fedavg_uniform_class_weighted"].append(
            float(
                evaluate(
                    weighted_uniform_model,
                    test_loader,
                    device,
                    binarize=False,
                    hard=True,
                    flat=True,
                )
            )
        )
        weighted_sample_model = copy.deepcopy(base_model).to(device)
        set_gate_weights(
            weighted_sample_model,
            _weighted_gate_weights(weighted_client_models, client_sizes),
        )
        raw["fedavg_weighted_class_weighted"].append(
            float(
                evaluate(
                    weighted_sample_model,
                    test_loader,
                    device,
                    binarize=False,
                    hard=True,
                    flat=True,
                )
            )
        )
        weighted_central_model = copy.deepcopy(base_model).to(device)
        _train_class_weighted(
            weighted_central_model,
            _loader(
                torch.utils.data.ConcatDataset(train_datasets),
                shuffle=True,
            ),
            args.central_iters,
            class_weights,
            args.lr,
            device,
        )
        raw["centralized_class_weighted"].append(
            float(
                evaluate(
                    weighted_central_model,
                    test_loader,
                    device,
                    binarize=False,
                    hard=True,
                    flat=True,
                )
            )
        )

        seed_details[str(seed)] = {
            "client_details": client_details,
            "client_accuracies": client_accuracies,
            "pooled_train_class_counts": pooled_train_counts.tolist(),
            "class_weights": class_weights.tolist(),
            "diverse_comm": diverse_comm,
            "elapsed_sec": time.time() - seed_start,
        }
        payload = {
            "config": vars(args),
            "dataset": {
                "name": "ExtraSensory",
                "client_unit": "user",
                "class_names": LABEL_NAMES,
                "feature_columns": feature_columns,
                "train_users": sorted(train_ids),
                "test_users": sorted(test_ids),
                "thresholds": thresholds.tolist(),
                "test_original_class_counts": test_original_counts,
                "test_balanced_examples": len(test_dataset),
                "partition_audit": audit,
            },
            "raw": raw,
            "seed_details": seed_details,
            "summary": _summary(raw),
            "elapsed_sec": time.time() - total_start,
        }
        _save(output_path, payload)
        print(
            f"seed={seed} central={100 * raw['centralized'][-1]:.2f} "
            f"client={100 * raw['client_mean'][-1]:.2f} "
            f"FedAvg={100 * raw['fedavg_uniform'][-1]:.2f}/"
            f"{100 * raw['fedavg_weighted'][-1]:.2f} "
            f"ensemble={100 * raw['ensemble'][-1]:.2f} "
            f"FBFS={100 * raw['fbfs'][-1]:.2f}/"
            f"{100 * raw['diverse_fbfs'][-1]:.2f} "
            f"weighted-control central/FedAvg="
            f"{100 * raw['centralized_class_weighted'][-1]:.2f}/"
            f"{100 * raw['fedavg_uniform_class_weighted'][-1]:.2f}/"
            f"{100 * raw['fedavg_weighted_class_weighted'][-1]:.2f} "
            f"({(time.time() - seed_start) / 60:.1f} min)",
            flush=True,
        )

        del (
            client_models,
            fedavg_model,
            weighted_model,
            central_model,
            weighted_client_models,
            weighted_uniform_model,
            weighted_sample_model,
            weighted_central_model,
        )
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    output_path.with_suffix(".DONE").write_text("completed\n")


if __name__ == "__main__":
    main()
