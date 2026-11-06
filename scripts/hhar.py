"""Natural device-client evaluation on the HHAR sensor dataset.

The eight physical smartphones are federated clients. Two complete users are
held out for test, so adjacent windows and user-specific motion patterns cannot
cross the train/test boundary. Inputs are non-overlapping accelerometer windows
binarized with fixed physical thresholds; no pooled normalization is fitted.
"""

import argparse
import copy
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "federated"))
from common import (  # noqa: E402
    bfs_federated,
    bfs_federated_diverse,
    evaluate,
    evaluate_ensemble,
    fedavg_gate_weights,
    make_difflogic_net,
    set_gate_weights,
    train_one_client,
)


LABEL_ALIASES = {
    "bike": "bike",
    "biking": "bike",
    "sit": "sit",
    "sitting": "sit",
    "stand": "stand",
    "standing": "stand",
    "walk": "walk",
    "walking": "walk",
    "stairsup": "stairsup",
    "stairup": "stairsup",
    "stairs_up": "stairsup",
    "stairsdown": "stairsdown",
    "stairdown": "stairsdown",
    "stairs_down": "stairsdown",
}
LABELS = ["bike", "sit", "stand", "walk", "stairsup", "stairsdown"]


def _device(device):
    return str(device).strip()


def _label(value):
    normalized = str(value).strip().lower().replace(" ", "").replace("-", "")
    return LABEL_ALIASES.get(normalized)


def _window_bits(values, thresholds):
    values = np.asarray(values, dtype=np.float32)
    return np.concatenate(
        [(values > threshold).reshape(-1) for threshold in thresholds]
    ).astype(np.uint8)


def _reservoir_windows(frame, window_size, thresholds, max_per_endpoint_class, seed):
    """Create fixed-sample windows, retaining a bounded deterministic reservoir."""
    rng = np.random.default_rng(seed)
    reservoirs = {}
    seen = {}

    sort_columns = ["User", "Device", "Creation_Time"]
    frame = frame.sort_values(sort_columns, kind="stable")
    for (user, device), group in frame.groupby(["User", "Device"], sort=True):
        group = group.reset_index(drop=True)
        times = group["Creation_Time"].to_numpy(dtype=np.float64)
        labels = group["label"].to_numpy()
        positive_deltas = np.diff(times)
        positive_deltas = positive_deltas[positive_deltas > 0]
        typical_delta = float(np.median(positive_deltas)) if positive_deltas.size else 1.0
        gap_limit = 10.0 * typical_delta

        starts = np.zeros(len(group), dtype=bool)
        starts[0] = True
        if len(group) > 1:
            deltas = np.diff(times)
            starts[1:] = (
                (labels[1:] != labels[:-1])
                | (deltas <= 0)
                | (deltas > gap_limit)
            )
        run_ids = np.cumsum(starts)

        for _, run in group.groupby(run_ids, sort=False):
            label_name = run["label"].iloc[0]
            label_id = LABELS.index(label_name)
            values = run[["x", "y", "z"]].to_numpy(dtype=np.float32)
            usable = len(values) - (len(values) % window_size)
            for offset in range(0, usable, window_size):
                key = (str(user), _device(device), label_id)
                seen[key] = seen.get(key, 0) + 1
                item = _window_bits(
                    values[offset : offset + window_size], thresholds
                )
                bucket = reservoirs.setdefault(key, [])
                if len(bucket) < max_per_endpoint_class:
                    bucket.append(item)
                else:
                    replacement = int(rng.integers(0, seen[key]))
                    if replacement < max_per_endpoint_class:
                        bucket[replacement] = item

    xs, ys, users, devices = [], [], [], []
    for (user, device, label_id), windows in sorted(reservoirs.items()):
        xs.extend(windows)
        ys.extend([label_id] * len(windows))
        users.extend([user] * len(windows))
        devices.extend([device] * len(windows))
    return {
        "x": np.asarray(xs, dtype=np.uint8),
        "y": np.asarray(ys, dtype=np.int64),
        "user": np.asarray(users),
        "device": np.asarray(devices),
        "seen_per_endpoint_class": {
            "|".join(map(str, key)): int(value) for key, value in seen.items()
        },
    }


def preprocess(args):
    cache_path = Path(args.cache)
    if cache_path.is_file() and not args.rebuild_cache:
        cached = np.load(cache_path, allow_pickle=False)
        return {
            "x": cached["x"],
            "y": cached["y"],
            "user": cached["user"],
            "device": cached["device"],
        }

    usecols = ["Creation_Time", "x", "y", "z", "User", "Device", "gt"]
    print(f"Reading {args.csv}", flush=True)
    frame = pd.read_csv(
        args.csv,
        usecols=usecols,
        dtype={
            "Creation_Time": np.int64,
            "x": np.float32,
            "y": np.float32,
            "z": np.float32,
            "User": "category",
            "Device": "category",
            "gt": "category",
        },
        low_memory=False,
    )
    frame["label"] = frame["gt"].map(_label)
    frame = frame[frame["label"].notna()].copy()
    frame = frame.dropna(
        subset=["Creation_Time", "x", "y", "z", "User", "Device", "label"]
    )
    finite = np.isfinite(frame[["Creation_Time", "x", "y", "z"]]).all(axis=1)
    frame = frame[finite]
    print(
        f"Retained {len(frame):,} labeled finite accelerometer samples",
        flush=True,
    )

    data = _reservoir_windows(
        frame,
        args.window_size,
        args.thresholds,
        args.max_windows_per_endpoint_class,
        args.preprocess_seed,
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        x=data["x"],
        y=data["y"],
        user=data["user"],
        device=data["device"],
    )
    metadata_path = cache_path.with_suffix(".metadata.json")
    with metadata_path.open("w") as handle:
        json.dump(
            {
                "csv": args.csv,
                "window_size": args.window_size,
                "thresholds": args.thresholds,
                "max_windows_per_endpoint_class": (
                    args.max_windows_per_endpoint_class
                ),
                "labels": LABELS,
                "windows": int(len(data["y"])),
                "seen_per_endpoint_class": data["seen_per_endpoint_class"],
            },
            handle,
            indent=2,
        )
    print(f"Cached {len(data['y']):,} windows at {cache_path}", flush=True)
    return {key: data[key] for key in ["x", "y", "user", "device"]}


def _sample_indices(indices, labels, per_class, rng):
    sampled = []
    for class_id in range(len(LABELS)):
        class_indices = indices[labels[indices] == class_id].copy()
        rng.shuffle(class_indices)
        sampled.extend(class_indices[:per_class].tolist())
    rng.shuffle(sampled)
    return sampled


def _stratified_split(indices, labels, holdout_fraction, rng):
    train, holdout = [], []
    indices = np.asarray(indices)
    for class_id in range(len(LABELS)):
        class_indices = indices[labels[indices] == class_id].copy()
        rng.shuffle(class_indices)
        count = max(1, int(round(holdout_fraction * len(class_indices))))
        count = min(count, max(1, len(class_indices) - 1))
        holdout.extend(class_indices[:count].tolist())
        train.extend(class_indices[count:].tolist())
    rng.shuffle(train)
    rng.shuffle(holdout)
    return train, holdout


def _loader(dataset, indices, batch_size=128, shuffle=False):
    subset = torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        subset,
        batch_size=min(batch_size, max(1, len(subset))),
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


def _summary(raw):
    return {
        method: {
            "mean": float(np.mean(values) * 100),
            "std": (
                float(np.std(values, ddof=1) * 100) if len(values) > 1 else 0.0
            ),
            "n": len(values),
        }
        for method, values in raw.items()
        if values
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=(
            "data-hhar/Activity recognition exp/"
            "Phones_accelerometer.csv"
        ),
    )
    parser.add_argument("--cache", default="data-hhar/phone_accel_windows.npz")
    parser.add_argument("--rebuild-cache", action="store_true")
    parser.add_argument("--preprocess-only", action="store_true")
    parser.add_argument("--preprocess-seed", type=int, default=20260727)
    parser.add_argument("--window-size", type=int, default=128)
    parser.add_argument(
        "--thresholds", nargs="+", type=float, default=[-5.0, 0.0, 5.0]
    )
    parser.add_argument(
        "--max-windows-per-endpoint-class", type=int, default=500
    )
    parser.add_argument("--test-users", nargs="+", default=None)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46])
    parser.add_argument("--windows-per-client-class", type=int, default=300)
    parser.add_argument("--test-windows-per-class", type=int, default=1000)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--neurons", type=int, default=900)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--num-iters", type=int, default=800)
    parser.add_argument("--central-iters", type=int, default=5000)
    parser.add_argument("--num-select", type=int, default=100)
    parser.add_argument("--shortlist-per-class", type=int, default=200)
    parser.add_argument("--diversity-lambda", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--out", default="scripts/results/rebuttal_natural_hhar.json"
    )
    args = parser.parse_args()

    data = preprocess(args)
    if args.preprocess_only:
        return

    users = sorted(np.unique(data["user"]).tolist())
    devices = sorted(np.unique(data["device"]).tolist())
    test_users = args.test_users or users[-2:]
    train_users = [user for user in users if user not in test_users]
    if not train_users or any(user not in users for user in test_users):
        raise ValueError(f"Invalid test users {test_users}; available users are {users}")

    x = torch.from_numpy(data["x"].astype(np.float32))
    y = torch.from_numpy(data["y"])
    dataset = torch.utils.data.TensorDataset(x, y)
    labels = data["y"]
    train_user_mask = np.isin(data["user"], train_users)
    test_user_mask = np.isin(data["user"], test_users)

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(
        f"HHAR: device={device}, users={users}, devices={devices}, "
        f"train_users={train_users}, test_users={test_users}, "
        f"input_dim={x.shape[1]}",
        flush=True,
    )

    methods = [
        "centralized",
        "client_mean",
        "fedavg_uniform",
        "fedavg_weighted",
        "ensemble",
        "fbfs",
        "diverse_fbfs",
    ]
    raw = {method: [] for method in methods}
    seed_details = {}
    total_start = time.time()

    for seed in args.seeds:
        seed_start = time.time()
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        rng = np.random.default_rng(seed)

        client_train_indices, client_selection_indices = [], []
        active_devices = []
        for device_name in devices:
            candidates = np.flatnonzero(
                train_user_mask & (data["device"] == device_name)
            )
            sampled = _sample_indices(
                candidates,
                labels,
                args.windows_per_client_class,
                rng,
            )
            if len(set(labels[sampled].tolist())) != len(LABELS):
                print(f"Skipping incomplete device client {device_name}", flush=True)
                continue
            train_indices, selection_indices = _stratified_split(
                sampled, labels, args.holdout_frac, rng
            )
            active_devices.append(device_name)
            client_train_indices.append(train_indices)
            client_selection_indices.append(selection_indices)

        test_candidates = np.flatnonzero(test_user_mask)
        test_indices = _sample_indices(
            test_candidates,
            labels,
            args.test_windows_per_class,
            rng,
        )
        train_loaders = [
            _loader(dataset, indices, shuffle=True)
            for indices in client_train_indices
        ]
        selection_loaders = [
            _loader(dataset, indices) for indices in client_selection_indices
        ]
        test_loader = _loader(dataset, test_indices, batch_size=512)
        client_sizes = [len(indices) for indices in client_train_indices]

        base_model = make_difflogic_net(
            x.shape[1], args.neurons, args.layers, len(LABELS), tau=10.0
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
            print(
                f"  seed={seed} client={client_id + 1}/{len(train_loaders)} "
                f"({active_devices[client_id]}) trained",
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
            weighted_model, _weighted_gate_weights(client_models, client_sizes)
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
            num_classes=len(LABELS),
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
            num_classes=len(LABELS),
            num_select=args.num_select,
            binarize=False,
            flat=True,
            diversity_lambda=args.diversity_lambda,
            shortlist_per_class=args.shortlist_per_class,
            return_comm=True,
        )
        raw["diverse_fbfs"].append(float(diverse_accuracy))

        central_indices = [
            index for indices in client_train_indices for index in indices
        ]
        central_model = copy.deepcopy(base_model).to(device)
        train_one_client(
            central_model,
            _loader(dataset, central_indices, shuffle=True),
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

        seed_details[str(seed)] = {
            "active_devices": active_devices,
            "client_train_sizes": client_sizes,
            "client_selection_sizes": [
                len(indices) for indices in client_selection_indices
            ],
            "test_examples": len(test_indices),
            "client_accuracies": client_accuracies,
            "diverse_comm": {
                key: int(value) for key, value in diverse_comm.items()
            },
            "elapsed_sec": time.time() - seed_start,
        }
        payload = {
            "config": vars(args),
            "dataset": {
                "source": "UCI HHAR, DOI 10.24432/C5689X",
                "client_unit": "physical smartphone",
                "users": users,
                "train_users": train_users,
                "test_users": test_users,
                "devices": devices,
                "labels": LABELS,
                "input_dim": int(x.shape[1]),
                "cached_windows": len(dataset),
            },
            "raw": raw,
            "seed_details": seed_details,
            "partial_summary": _summary(raw),
            "elapsed_sec": time.time() - total_start,
        }
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(
            f"seed={seed}: central={100 * raw['centralized'][-1]:.2f} "
            f"FedAvg={100 * raw['fedavg_uniform'][-1]:.2f}/"
            f"{100 * raw['fedavg_weighted'][-1]:.2f} "
            f"ensemble={100 * raw['ensemble'][-1]:.2f} "
            f"FBFS={100 * fbfs_accuracy:.2f} "
            f"diverse={100 * diverse_accuracy:.2f} "
            f"({(time.time() - seed_start) / 60:.1f} min)",
            flush=True,
        )

        for model in client_models:
            model.cpu()
        del (
            client_models,
            fedavg_model,
            weighted_model,
            central_model,
        )
        if device.type == "mps":
            torch.mps.empty_cache()
        elif device.type == "cuda":
            torch.cuda.empty_cache()

    payload["summary"] = _summary(raw)
    payload.pop("partial_summary", None)
    payload["elapsed_sec"] = time.time() - total_start
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    with open(args.out.rsplit(".", 1)[0] + ".DONE", "w") as handle:
        handle.write(f"completed in {(time.time() - total_start) / 60:.1f} min\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
