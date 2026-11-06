"""Natural-writer FBFS evaluation on the official TFF Federated EMNIST data.

The experiment samples a fixed cohort before training from writers meeting a
small data-availability threshold. Each writer is a client. Local training data
are split into train/selection subsets, while TFF's official per-writer test
split is reserved for evaluation.
"""

import argparse
import copy
import json
import os
import random
import sys
import tarfile
import time
import urllib.request

import h5py
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "federated"))
from common import (  # noqa: E402
    evaluate,
    evaluate_ensemble,
    fedavg_gate_softconf,
    fedavg_gate_weights,
    get_last_layer_outputs,
    make_difflogic_net,
    set_gate_weights,
    train_one_client,
)


VARIANTS = {
    "digits": {
        "archive": "fed_emnist_digitsonly.tar.bz2",
        "prefix": "fed_emnist_digitsonly",
        "num_classes": 10,
        "neurons": 500,
        "shortlist_per_class": 200,
    },
    "full": {
        "archive": "fed_emnist.tar.bz2",
        "prefix": "fed_emnist",
        "num_classes": 62,
        "neurons": 1240,
        "shortlist_per_class": 100,
    },
}
TFF_DATA_ROOT = "https://storage.googleapis.com/tff-datasets-public"


def _download_and_extract(data_dir, variant):
    config = VARIANTS[variant]
    train_path = os.path.join(data_dir, f"{config['prefix']}_train.h5")
    test_path = os.path.join(data_dir, f"{config['prefix']}_test.h5")
    if os.path.isfile(train_path) and os.path.isfile(test_path):
        return train_path, test_path

    os.makedirs(data_dir, exist_ok=True)
    archive_path = os.path.join(data_dir, config["archive"])
    if not os.path.isfile(archive_path):
        url = f"{TFF_DATA_ROOT}/{config['archive']}"
        print(f"Downloading {url}", flush=True)
        urllib.request.urlretrieve(url, archive_path)
    with tarfile.open(archive_path, "r:bz2") as archive:
        archive.extractall(data_dir, filter="data")
    if not os.path.isfile(train_path) or not os.path.isfile(test_path):
        raise RuntimeError(f"Unexpected Federated EMNIST archive under {data_dir}")
    return train_path, test_path


def _choose_cohort(
    train_path,
    test_path,
    cohort_size,
    cohort_seed,
    min_train_examples,
    min_test_examples,
):
    with h5py.File(train_path, "r") as train_h5, h5py.File(
        test_path, "r"
    ) as test_h5:
        train_group = train_h5["examples"]
        test_group = test_h5["examples"]
        common_ids = sorted(set(train_group) & set(test_group))
        eligible = [
            client_id
            for client_id in common_ids
            if len(train_group[client_id]["label"]) >= min_train_examples
            and len(test_group[client_id]["label"]) >= min_test_examples
        ]
    if len(eligible) < cohort_size:
        raise ValueError(
            f"Only {len(eligible)} writers meet the eligibility criteria"
        )
    rng = np.random.default_rng(cohort_seed)
    selected = rng.choice(eligible, cohort_size, replace=False).tolist()
    return sorted(selected), len(common_ids), len(eligible)


def _tensor_dataset(pixels, labels):
    # TFF uses one for background and zero for ink; invert to match MNIST.
    x = torch.from_numpy(1.0 - np.asarray(pixels, dtype=np.float32))
    y = torch.from_numpy(np.asarray(labels, dtype=np.int64))
    return torch.utils.data.TensorDataset(x, y)


def _stratified_split(labels, holdout_frac, seed):
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)
    train_indices, selection_indices = [], []
    for class_id in sorted(set(labels.tolist())):
        indices = np.flatnonzero(labels == class_id)
        rng.shuffle(indices)
        n_selection = 0
        if len(indices) > 1:
            n_selection = min(
                max(1, int(round(holdout_frac * len(indices)))),
                len(indices) - 1,
            )
        selection_indices.extend(indices[:n_selection].tolist())
        train_indices.extend(indices[n_selection:].tolist())
    rng.shuffle(train_indices)
    rng.shuffle(selection_indices)
    return train_indices, selection_indices


def _loader(dataset, indices=None, batch_size=128, shuffle=False):
    selected = (
        dataset
        if indices is None
        else torch.utils.data.Subset(dataset, indices)
    )
    return torch.utils.data.DataLoader(
        selected,
        batch_size=min(batch_size, max(1, len(selected))),
        shuffle=shuffle,
        drop_last=False,
    )


def _load_cohort(
    train_path,
    test_path,
    client_ids,
    holdout_frac,
    split_seed,
    num_classes,
    min_federation_selection_per_class,
):
    full_train_datasets, test_datasets = [], []
    labels_by_client = []
    train_indices_by_client = []
    selection_indices_by_client = []
    with h5py.File(train_path, "r") as train_h5, h5py.File(
        test_path, "r"
    ) as test_h5:
        for client_index, client_id in enumerate(client_ids):
            train_group = train_h5["examples"][client_id]
            test_group = test_h5["examples"][client_id]
            train_labels = train_group["label"][:]
            train_dataset = _tensor_dataset(
                train_group["pixels"][:], train_labels
            )
            test_dataset = _tensor_dataset(
                test_group["pixels"][:], test_group["label"][:]
            )
            train_indices, selection_indices = _stratified_split(
                train_labels,
                holdout_frac,
                split_seed * 1000 + client_index,
            )
            full_train_datasets.append(train_dataset)
            test_datasets.append(test_dataset)
            labels_by_client.append(train_labels)
            train_indices_by_client.append(train_indices)
            selection_indices_by_client.append(selection_indices)

    selection_counts_before = np.sum(
        [
            np.bincount(
                labels[indices],
                minlength=num_classes,
            )
            for labels, indices in zip(
                labels_by_client, selection_indices_by_client
            )
        ],
        axis=0,
    )
    selection_counts = selection_counts_before.copy()
    rng = np.random.default_rng(split_seed + 20260727)
    moved = []
    for class_id in range(num_classes):
        while selection_counts[class_id] < min_federation_selection_per_class:
            candidates = []
            for client_index, (labels, train_indices, selection_indices) in enumerate(
                zip(
                    labels_by_client,
                    train_indices_by_client,
                    selection_indices_by_client,
                )
            ):
                matching = [
                    index
                    for index in train_indices
                    if labels[index] == class_id
                ]
                if matching and len(train_indices) > 1:
                    candidates.append(
                        (
                            len(matching),
                            len(train_indices),
                            -len(selection_indices),
                            float(rng.random()),
                            client_index,
                            matching,
                        )
                    )
            if not candidates:
                raise RuntimeError(
                    f"Cannot provide selection coverage for class {class_id}"
                )
            *_, client_index, matching = max(candidates)
            selected_index = int(rng.choice(matching))
            train_indices_by_client[client_index].remove(selected_index)
            selection_indices_by_client[client_index].append(selected_index)
            selection_counts[class_id] += 1
            moved.append(
                {
                    "class_id": class_id,
                    "client_id": client_ids[client_index],
                    "example_index": selected_index,
                }
            )

    train_loaders = [
        _loader(dataset, indices, shuffle=True)
        for dataset, indices in zip(
            full_train_datasets, train_indices_by_client
        )
    ]
    selection_loaders = [
        _loader(dataset, indices, shuffle=False)
        for dataset, indices in zip(
            full_train_datasets, selection_indices_by_client
        )
    ]
    client_details = []
    for client_id, train_dataset, test_dataset, labels, train_indices, selection_indices in zip(
        client_ids,
        full_train_datasets,
        test_datasets,
        labels_by_client,
        train_indices_by_client,
        selection_indices_by_client,
    ):
        client_details.append(
            {
                "client_id": client_id,
                "official_train_examples": len(train_dataset),
                "local_train_examples": len(train_indices),
                "selection_examples": len(selection_indices),
                "official_test_examples": len(test_dataset),
                "selection_class_counts": np.bincount(
                    labels[selection_indices],
                    minlength=num_classes,
                ).tolist(),
            }
        )
    full_train = torch.utils.data.ConcatDataset(full_train_datasets)
    test = torch.utils.data.ConcatDataset(test_datasets)
    return (
        train_loaders,
        selection_loaders,
        _loader(full_train, shuffle=True),
        _loader(test, batch_size=512),
        client_details,
        {
            "minimum_per_class": min_federation_selection_per_class,
            "counts_before_repair": selection_counts_before.tolist(),
            "counts_after_repair": selection_counts.tolist(),
            "moved_examples": moved,
        },
    )


def _pooled_scores(models, selection_loaders, device, num_classes):
    per_client_gates = []
    in_sum = None
    in_count = torch.zeros(num_classes)
    total_sum = None
    total_count = 0
    for client_index, loader in enumerate(selection_loaders):
        gates = []
        labels = None
        for model in models:
            local_gates, labels = get_last_layer_outputs(
                model, loader, device, binarize=True, flat=True
            )
            gates.append(local_gates)
        client_gates = torch.cat(gates, dim=1).float()
        labels = labels.cpu()
        per_client_gates.append(client_gates)
        if in_sum is None:
            num_candidates = client_gates.shape[1]
            in_sum = torch.zeros(num_classes, num_candidates)
            total_sum = torch.zeros(num_candidates)
        for class_id in range(num_classes):
            mask = labels == class_id
            if mask.any():
                in_sum[class_id] += client_gates[mask].sum(0)
                in_count[class_id] += mask.sum()
        total_sum += client_gates.sum(0)
        total_count += len(labels)
        if (client_index + 1) % 10 == 0:
            print(
                f"    selection aggregates {client_index + 1}/{len(selection_loaders)}",
                flush=True,
            )

    out_sum = total_sum.unsqueeze(0) - in_sum
    out_count = total_count - in_count
    scores = torch.zeros_like(in_sum)
    for class_id in range(num_classes):
        if in_count[class_id] > 0 and out_count[class_id] > 0:
            scores[class_id] = (
                in_sum[class_id] / in_count[class_id]
                - out_sum[class_id] / out_count[class_id]
            )
    return scores, per_client_gates, in_count


def _diverse_selection(
    scores,
    per_client_gates,
    device,
    num_select,
    shortlist_per_class,
    diversity_lambda,
):
    num_classes, num_candidates = scores.shape
    shortlist_set = set()
    shortlist_width = min(shortlist_per_class, num_candidates)
    for class_id in range(num_classes):
        shortlist_set.update(
            scores[class_id]
            .argsort(descending=True)[:shortlist_width]
            .tolist()
        )
    shortlist = sorted(shortlist_set)
    shortlist_indices = torch.tensor(shortlist, dtype=torch.long)
    shortlist_size = len(shortlist)

    pair_sum = torch.zeros(
        shortlist_size, shortlist_size, device=device, dtype=torch.float32
    )
    single_sum = torch.zeros(shortlist_size, device=device)
    total_count = 0
    for client_index, client_gates in enumerate(per_client_gates):
        local = client_gates[:, shortlist_indices].to(device)
        pair_sum.addmm_(local.T, local)
        single_sum += local.sum(0)
        total_count += local.shape[0]
        if (client_index + 1) % 20 == 0:
            print(
                f"    second moments {client_index + 1}/{len(per_client_gates)}",
                flush=True,
            )

    means = single_sum / total_count
    covariance = pair_sum / total_count - means[:, None] * means[None, :]
    variance = (means - means.square()).clamp(min=1e-8)
    correlation = (
        covariance / (variance.sqrt()[:, None] * variance.sqrt()[None, :])
    ).abs().cpu()
    del pair_sum, single_sum, covariance, variance
    if device.type == "cuda":
        torch.cuda.empty_cache()

    shortlisted_scores = scores[:, shortlist_indices]
    selected = {}
    for class_id in range(num_classes):
        selected_local = []
        max_correlation = torch.zeros(shortlist_size)
        for _ in range(num_select):
            adjusted = (
                shortlisted_scores[class_id]
                - diversity_lambda * max_correlation
            )
            if selected_local:
                adjusted[selected_local] = -float("inf")
            best = int(adjusted.argmax())
            selected_local.append(best)
            max_correlation = torch.maximum(
                max_correlation, correlation[:, best]
            )
        selected[class_id] = shortlist_indices[
            torch.tensor(selected_local, dtype=torch.long)
        ]
    return selected, shortlist_size


def _evaluate_selected(
    models,
    test_loader,
    device,
    selections,
    gates_per_model,
    num_classes,
):
    logits = {
        name: torch.zeros(len(test_loader.dataset), num_classes)
        for name in selections
    }
    labels = None
    for model_index, model in enumerate(models):
        gates, labels = get_last_layer_outputs(
            model, test_loader, device, binarize=True, flat=True
        )
        start = model_index * gates_per_model
        end = start + gates_per_model
        for method, selected_by_class in selections.items():
            for class_id, selected in selected_by_class.items():
                local = selected[(selected >= start) & (selected < end)] - start
                if len(local):
                    logits[method][:, class_id] += gates[:, local].sum(1)
        if (model_index + 1) % 20 == 0:
            print(
                f"    selected-circuit eval {model_index + 1}/{len(models)}",
                flush=True,
            )
    return {
        method: float((values.argmax(1) == labels).float().mean())
        for method, values in logits.items()
    }


def _run_fbfs(
    models,
    selection_loaders,
    test_loader,
    device,
    num_classes,
    num_select,
    shortlist_per_class,
    diversity_lambda,
):
    scores, per_client_gates, class_counts = _pooled_scores(
        models, selection_loaders, device, num_classes
    )
    standard = {
        class_id: scores[class_id]
        .argsort(descending=True)[:num_select]
        for class_id in range(num_classes)
    }
    diverse, shortlist_size = _diverse_selection(
        scores,
        per_client_gates,
        device,
        num_select,
        shortlist_per_class,
        diversity_lambda,
    )
    gates_per_model = per_client_gates[0].shape[1] // len(models)
    del per_client_gates
    accuracies = _evaluate_selected(
        models,
        test_loader,
        device,
        {"fbfs": standard, "diverse_fbfs": diverse},
        gates_per_model,
        num_classes,
    )

    model_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in models[0].parameters()
    )
    num_candidates = len(models) * gates_per_model
    communication = {
        "model_exchange_bytes_per_client": len(models) * model_bytes,
        "round1_stats_bytes_per_client": (
            num_classes * num_candidates * 4 + num_classes * 8
        ),
        "round2_stats_bytes_per_client": (
            shortlist_size * shortlist_size * 4 + shortlist_size * 4
        ),
        "shortlist_size": shortlist_size,
        "num_candidates": num_candidates,
    }
    return accuracies, communication, class_counts.tolist()


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
    parser.add_argument("--variant", choices=sorted(VARIANTS), default="digits")
    parser.add_argument("--seeds", nargs="+", type=int, default=list(range(42, 47)))
    parser.add_argument("--cohort-size", type=int, default=100)
    parser.add_argument("--cohort-seed", type=int, default=20260727)
    parser.add_argument("--min-train-examples", type=int, default=40)
    parser.add_argument("--min-test-examples", type=int, default=5)
    parser.add_argument(
        "--min-federation-selection-per-class", type=int, default=10
    )
    parser.add_argument("--holdout-frac", type=float, default=0.2)
    parser.add_argument("--neurons", type=int)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--num-iters", type=int, default=500)
    parser.add_argument("--central-iters", type=int, default=5000)
    parser.add_argument("--num-select", type=int, default=100)
    parser.add_argument("--shortlist-per-class", type=int)
    parser.add_argument("--diversity-lambda", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data-dir", default="data-femnist")
    parser.add_argument("--out")
    args = parser.parse_args()

    variant = VARIANTS[args.variant]
    num_classes = variant["num_classes"]
    neurons = args.neurons or variant["neurons"]
    shortlist_per_class = (
        args.shortlist_per_class or variant["shortlist_per_class"]
    )
    if neurons % num_classes:
        raise ValueError("--neurons must be divisible by the class count")
    if args.num_select > args.cohort_size * (neurons // num_classes):
        raise ValueError("--num-select exceeds the per-class candidate pool")
    if args.out is None:
        args.out = f"scripts/results/rebuttal_femnist_{args.variant}.json"

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}", flush=True)

    train_path, test_path = _download_and_extract(args.data_dir, args.variant)
    cohort, total_writers, eligible_writers = _choose_cohort(
        train_path,
        test_path,
        args.cohort_size,
        args.cohort_seed,
        args.min_train_examples,
        args.min_test_examples,
    )
    print(
        f"FEMNIST-{args.variant}: fixed cohort {len(cohort)} from "
        f"{eligible_writers}/{total_writers} eligible writers",
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
        (
            train_loaders,
            selection_loaders,
            central_loader,
            test_loader,
            client_details,
            selection_coverage,
        ) = _load_cohort(
            train_path,
            test_path,
            cohort,
            args.holdout_frac,
            seed,
            num_classes,
            args.min_federation_selection_per_class,
        )
        base_model = make_difflogic_net(
            784, neurons, args.layers, num_classes, tau=10.0
        )
        models = []
        for client_index, loader in enumerate(train_loaders):
            model = copy.deepcopy(base_model).to(device)
            train_one_client(
                model,
                loader,
                args.num_iters,
                lr=args.lr,
                device=device,
                binarize=True,
                flat=True,
            )
            models.append(model)
            if (client_index + 1) % 10 == 0:
                print(
                    f"  seed={seed} clients trained "
                    f"{client_index + 1}/{len(train_loaders)}",
                    flush=True,
                )

        client_accuracies = [
            evaluate(
                model,
                test_loader,
                device,
                binarize=True,
                hard=True,
                flat=True,
            )
            for model in models
        ]
        raw["client_mean"].append(float(np.mean(client_accuracies)))
        raw["ensemble"].append(
            float(
                evaluate_ensemble(
                    models,
                    test_loader,
                    device,
                    binarize=True,
                    hard=True,
                    flat=True,
                )
            )
        )

        fedavg_model = copy.deepcopy(base_model).to(device)
        set_gate_weights(fedavg_model, fedavg_gate_weights(models))
        raw["fedavg"].append(
            float(
                evaluate(
                    fedavg_model,
                    test_loader,
                    device,
                    binarize=True,
                    hard=True,
                    flat=True,
                )
            )
        )
        softconf_model = copy.deepcopy(base_model).to(device)
        set_gate_weights(softconf_model, fedavg_gate_softconf(models))
        raw["softconf"].append(
            float(
                evaluate(
                    softconf_model,
                    test_loader,
                    device,
                    binarize=True,
                    hard=True,
                    flat=True,
                )
            )
        )

        fbfs_accuracies, communication, class_counts = _run_fbfs(
            models,
            selection_loaders,
            test_loader,
            device,
            num_classes,
            args.num_select,
            shortlist_per_class,
            args.diversity_lambda,
        )
        for method, accuracy in fbfs_accuracies.items():
            raw[method].append(accuracy)

        central_model = copy.deepcopy(base_model).to(device)
        train_one_client(
            central_model,
            central_loader,
            args.central_iters,
            lr=args.lr,
            device=device,
            binarize=True,
            flat=True,
        )
        raw["centralized"].append(
            float(
                evaluate(
                    central_model,
                    test_loader,
                    device,
                    binarize=True,
                    hard=True,
                    flat=True,
                )
            )
        )

        seed_details[str(seed)] = {
            "client_details": client_details,
            "federation_selection_class_counts": class_counts,
            "selection_coverage": selection_coverage,
            "communication": communication,
            "elapsed_sec": time.time() - seed_start,
        }
        payload = {
            "config": {
                **vars(args),
                "num_classes": num_classes,
                "neurons": neurons,
                "shortlist_per_class": shortlist_per_class,
            },
            "dataset": {
                "source": "TFF Federated EMNIST",
                "variant": args.variant,
                "total_writers": total_writers,
                "eligible_writers": eligible_writers,
                "cohort": cohort,
                "cohort_official_train_examples": sum(
                    detail["official_train_examples"]
                    for detail in client_details
                ),
                "cohort_local_train_examples": sum(
                    detail["local_train_examples"] for detail in client_details
                ),
                "cohort_selection_examples": sum(
                    detail["selection_examples"] for detail in client_details
                ),
                "cohort_official_test_examples": sum(
                    detail["official_test_examples"] for detail in client_details
                ),
            },
            "raw": raw,
            "seed_details": seed_details,
            "partial_summary": _summary(raw),
            "elapsed_sec": time.time() - start,
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(
            f"seed={seed} central={100 * raw['centralized'][-1]:.1f} "
            f"ensemble={100 * raw['ensemble'][-1]:.1f} "
            f"FBFS={100 * raw['fbfs'][-1]:.1f} "
            f"diverse={100 * raw['diverse_fbfs'][-1]:.1f} "
            f"({(time.time() - seed_start) / 60:.1f} min)",
            flush=True,
        )

        for model in models:
            model.cpu()
        del models, fedavg_model, softconf_model, central_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    payload["summary"] = _summary(raw)
    payload.pop("partial_summary", None)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    with open(args.out.rsplit(".", 1)[0] + ".DONE", "w") as handle:
        handle.write(
            f"completed {len(args.seeds)} seeds in "
            f"{(time.time() - start) / 60:.1f} min\n"
        )
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
