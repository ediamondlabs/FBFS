"""Natural-client FBFS evaluation on UCI HAR.

The official split contains 21 training subjects and 9 held-out test subjects.
Each training subject is one federated client. Continuous features are converted
to Boolean inputs with fixed thresholds, avoiding global preprocessing
statistics. Results are persisted after every seed for preemption safety.
"""

import argparse
import copy
import json
import os
import random
import sys
import time
import urllib.request
import zipfile

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "federated"))
from common import (  # noqa: E402
    bfs_federated,
    bfs_federated_diverse,
    evaluate,
    evaluate_ensemble,
    fedavg_gate_softconf,
    fedavg_gate_weights,
    make_difflogic_net,
    set_gate_weights,
    train_one_client,
)


UCI_HAR_URL = (
    "https://archive.ics.uci.edu/static/public/240/"
    "human+activity+recognition+using+smartphones.zip"
)


def _download_har(data_dir):
    root = os.path.join(data_dir, "UCI HAR Dataset")
    if os.path.isfile(os.path.join(root, "train", "X_train.txt")):
        return root
    os.makedirs(data_dir, exist_ok=True)
    archive = os.path.join(data_dir, "uci_har.zip")
    if not os.path.isfile(archive):
        print(f"Downloading {UCI_HAR_URL}", flush=True)
        urllib.request.urlretrieve(UCI_HAR_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        zf.extractall(data_dir)
    nested_archive = os.path.join(data_dir, "UCI HAR Dataset.zip")
    if os.path.isfile(nested_archive):
        with zipfile.ZipFile(nested_archive) as zf:
            zf.extractall(data_dir)
    if not os.path.isfile(os.path.join(root, "train", "X_train.txt")):
        raise RuntimeError(f"Unexpected UCI HAR archive layout under {data_dir}")
    return root


def _load_split(root, split, thresholds):
    split_dir = os.path.join(root, split)
    x = np.loadtxt(os.path.join(split_dir, f"X_{split}.txt"), dtype=np.float32)
    y = np.loadtxt(os.path.join(split_dir, f"y_{split}.txt"), dtype=np.int64) - 1
    subjects = np.loadtxt(
        os.path.join(split_dir, f"subject_{split}.txt"), dtype=np.int64
    )
    bits = np.concatenate([(x > threshold).astype(np.float32) for threshold in thresholds], axis=1)
    return torch.from_numpy(bits), torch.from_numpy(y), subjects


def _stratified_subject_split(indices, labels, holdout_frac, seed):
    train, holdout = [], []
    rng = np.random.default_rng(seed)
    labels_np = labels.numpy()
    for class_id in sorted(set(labels_np[indices].tolist())):
        class_indices = np.asarray(indices)[labels_np[indices] == class_id].copy()
        rng.shuffle(class_indices)
        n_holdout = max(1, int(round(holdout_frac * len(class_indices))))
        if len(class_indices) > 1:
            n_holdout = min(n_holdout, len(class_indices) - 1)
        holdout.extend(class_indices[:n_holdout].tolist())
        train.extend(class_indices[n_holdout:].tolist())
    rng.shuffle(train)
    rng.shuffle(holdout)
    return train, holdout


def _loader(dataset, indices=None, batch_size=128, shuffle=False):
    selected = dataset if indices is None else torch.utils.data.Subset(dataset, indices)
    return torch.utils.data.DataLoader(
        selected,
        batch_size=min(batch_size, max(1, len(selected))),
        shuffle=shuffle,
        drop_last=False,
    )


def _summary(raw):
    return {
        method: {
            "mean": float(np.mean(values) * 100),
            "std": float(np.std(values, ddof=1) * 100) if len(values) > 1 else 0.0,
        }
        for method, values in raw.items()
        if values
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seeds", nargs="+", type=int, default=list(range(42, 57))
    )
    parser.add_argument("--data-dir", default="data-uci-har")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[-0.5, 0.0, 0.5])
    parser.add_argument("--neurons", type=int, default=900)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--num-iters", type=int, default=800)
    parser.add_argument("--central-iters", type=int, default=5000)
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--num-select", type=int, default=100)
    parser.add_argument("--shortlist-per-class", type=int, default=200)
    parser.add_argument("--diversity-lambda", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--out", default="scripts/results/rebuttal_natural_har.json"
    )
    args = parser.parse_args()

    if args.neurons % 6:
        raise ValueError("--neurons must be divisible by the six HAR classes")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}", flush=True)

    root = _download_har(args.data_dir)
    train_x, train_y, train_subjects = _load_split(root, "train", args.thresholds)
    test_x, test_y, test_subjects = _load_split(root, "test", args.thresholds)
    train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
    test_loader = _loader(test_dataset, batch_size=512)
    subjects = sorted(set(train_subjects.tolist()))
    print(
        f"HAR: {len(subjects)} natural clients, {len(train_dataset)} train, "
        f"{len(test_dataset)} test from {len(set(test_subjects.tolist()))} held-out subjects, "
        f"{train_x.shape[1]} Boolean inputs",
        flush=True,
    )

    methods = [
        "centralized",
        "client_mean",
        "fedavg",
        "softconf",
        "ensemble",
        "fbfs",
        "diverse_fbfs",
    ]
    raw = {method: [] for method in methods}
    seed_details = {}
    start = time.time()

    for seed in args.seeds:
        seed_start = time.time()
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)

        base_model = make_difflogic_net(
            train_x.shape[1], args.neurons, args.layers, 6, tau=10.0
        )
        train_loaders, selection_loaders = [], []
        client_sizes, selection_class_counts = [], []
        for subject in subjects:
            subject_indices = np.where(train_subjects == subject)[0].tolist()
            train_indices, selection_indices = _stratified_subject_split(
                subject_indices,
                train_y,
                args.holdout_frac,
                seed * 1000 + subject,
            )
            train_loaders.append(_loader(train_dataset, train_indices, shuffle=True))
            selection_loaders.append(_loader(train_dataset, selection_indices))
            client_sizes.append(len(train_indices))
            selection_class_counts.append(
                torch.bincount(train_y[selection_indices], minlength=6).tolist()
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
                f"  seed={seed} client={client_id + 1}/{len(subjects)} trained",
                flush=True,
            )

        client_accs = [
            evaluate(model, test_loader, device, binarize=False, hard=True, flat=True)
            for model in client_models
        ]
        raw["client_mean"].append(float(np.mean(client_accs)))
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
        raw["fedavg"].append(
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
        softconf_model = copy.deepcopy(base_model).to(device)
        set_gate_weights(softconf_model, fedavg_gate_softconf(client_models))
        raw["softconf"].append(
            float(
                evaluate(
                    softconf_model,
                    test_loader,
                    device,
                    binarize=False,
                    hard=True,
                    flat=True,
                )
            )
        )

        fbfs_acc, _ = bfs_federated(
            client_models,
            selection_loaders,
            test_loader,
            device,
            num_classes=6,
            num_select=args.num_select,
            binarize=False,
            flat=True,
        )
        raw["fbfs"].append(float(fbfs_acc))
        diverse_acc, _, diverse_comm = bfs_federated_diverse(
            client_models,
            selection_loaders,
            test_loader,
            device,
            num_classes=6,
            num_select=args.num_select,
            binarize=False,
            flat=True,
            diversity_lambda=args.diversity_lambda,
            shortlist_per_class=args.shortlist_per_class,
            return_comm=True,
        )
        raw["diverse_fbfs"].append(float(diverse_acc))

        central_model = copy.deepcopy(base_model).to(device)
        central_loader = _loader(train_dataset, batch_size=128, shuffle=True)
        train_one_client(
            central_model,
            central_loader,
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
            "client_train_sizes": client_sizes,
            "selection_class_counts": selection_class_counts,
            "federation_selection_class_counts": np.sum(
                np.asarray(selection_class_counts), axis=0
            ).tolist(),
            "diverse_comm": {key: int(value) for key, value in diverse_comm.items()},
            "elapsed_sec": time.time() - seed_start,
        }
        payload = {
            "config": vars(args),
            "dataset": {
                "train_subjects": subjects,
                "test_subjects": sorted(set(test_subjects.tolist())),
                "train_examples": len(train_dataset),
                "test_examples": len(test_dataset),
                "boolean_input_dim": train_x.shape[1],
            },
            "raw": raw,
            "seed_details": seed_details,
            "partial_summary": _summary(raw),
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(
            f"seed={seed} central={100 * raw['centralized'][-1]:.1f} "
            f"ensemble={100 * raw['ensemble'][-1]:.1f} "
            f"FBFS={100 * fbfs_acc:.1f} diverse={100 * diverse_acc:.1f} "
            f"({(time.time() - seed_start) / 60:.1f} min)",
            flush=True,
        )

        for model in client_models:
            model.cpu()
        del client_models, fedavg_model, softconf_model, central_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload["summary"] = _summary(raw)
    payload["elapsed_sec"] = time.time() - start
    payload.pop("partial_summary", None)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    with open(args.out.rsplit(".", 1)[0] + ".DONE", "w") as handle:
        handle.write(f"completed in {(time.time() - start) / 60:.1f} min\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
