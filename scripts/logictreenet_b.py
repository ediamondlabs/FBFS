"""Multi-seed full-pool LogicTreeNet-B replication for the rebuttal.

This matches the reported centralized-probe protocol: five shared-wiring
clients, 6K local iterations, a fixed 1K CIFAR-10 probe, and all 1,638,400
last-layer candidates. It records selection time and peak CUDA memory and
persists results after every seed.
"""

import argparse
import copy
import json
import os
import random
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "federated"))
sys.path.insert(0, os.path.dirname(__file__))
from common import (  # noqa: E402
    evaluate,
    evaluate_ensemble,
    get_cifar10,
    make_conv_difflogic_net_tl,
    partition_dirichlet,
    train_one_client,
)
from full_pool_bfs import (  # noqa: E402
    collect_gates,
    diverse_bfs_streaming,
    standard_bfs,
)


def _checkpoint_dir(args, seed):
    if seed == 42 and args.seed42_checkpoint_dir:
        expected = [
            os.path.join(args.seed42_checkpoint_dir, f"client_{index}.pt")
            for index in range(args.num_clients)
        ]
        if all(os.path.isfile(path) for path in expected):
            return args.seed42_checkpoint_dir
    return os.path.join(
        args.checkpoint_root,
        f"paper_b_seed{seed}_a{args.alpha}",
        f"b1_cifar10_dirichlet_K{args.num_clients}",
    )


def _train_missing_clients(args, seed, train_dataset, device, checkpoint_dir):
    paths = [
        os.path.join(checkpoint_dir, f"client_{index}.pt")
        for index in range(args.num_clients)
    ]
    if all(os.path.isfile(path) for path in paths):
        print(f"seed={seed}: using existing checkpoints in {checkpoint_dir}", flush=True)
        return

    os.makedirs(checkpoint_dir, exist_ok=True)
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    client_indices = partition_dirichlet(
        train_dataset, args.num_clients, alpha=args.alpha, num_classes=10
    )
    base_model = make_conv_difflogic_net_tl(
        classes=10, depth=2, size="paper_b", device=device.type
    )
    for client_id, indices in enumerate(client_indices):
        path = paths[client_id]
        if os.path.isfile(path):
            print(f"  seed={seed} client={client_id}: checkpoint exists", flush=True)
            continue
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_dataset, indices),
            batch_size=min(args.batch_size, max(1, len(indices))),
            shuffle=True,
            drop_last=len(indices) >= 2 * args.batch_size,
        )
        model = copy.deepcopy(base_model).to(device)
        start = time.time()
        train_one_client(
            model,
            loader,
            args.num_iters,
            lr=args.lr,
            device=device,
            binarize=False,
            flat=False,
            optimizer_kind="adamw",
            weight_decay=args.weight_decay,
        )
        model.cpu()
        torch.save(model, path)
        del model
        torch.cuda.empty_cache()
        print(
            f"  seed={seed} client={client_id}: trained {args.num_iters} iters "
            f"in {(time.time() - start) / 60:.1f} min",
            flush=True,
        )
    base_model.cpu()
    del base_model
    torch.cuda.empty_cache()


def _client_references(checkpoint_dir, num_clients, test_loader, device):
    models = [
        torch.load(
            os.path.join(checkpoint_dir, f"client_{index}.pt"),
            map_location="cpu",
            weights_only=False,
        ).to(device)
        for index in range(num_clients)
    ]
    accuracies = [
        evaluate(
            model,
            test_loader,
            device,
            binarize=False,
            hard=True,
            flat=False,
        )
        for model in models
    ]
    ensemble = evaluate_ensemble(
        models,
        test_loader,
        device,
        binarize=False,
        hard=True,
        flat=False,
    )
    for model in models:
        model.cpu()
    del models
    torch.cuda.empty_cache()
    return accuracies, ensemble


def _summary(rows):
    metrics = [
        "client_mean",
        "ensemble",
        "bfs_full_pool",
        "diverse_bfs_full_pool",
    ]
    return {
        metric: {
            "mean": float(np.mean([row[metric] for row in rows]) * 100),
            "std": (
                float(
                    np.std([row[metric] for row in rows], ddof=1) * 100
                )
                if len(rows) > 1
                else 0.0
            ),
            "n": len(rows),
        }
        for metric in metrics
        if rows
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--num-clients", type=int, default=5)
    parser.add_argument("--num-iters", type=int, default=6000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.002)
    parser.add_argument("--probe-size", type=int, default=1000)
    parser.add_argument("--num-select", type=int, default=100)
    parser.add_argument("--diversity-lambda", type=float, default=0.3)
    parser.add_argument("--checkpoint-root", default="checkpoints/rebuttal")
    parser.add_argument(
        "--seed42-checkpoint-dir",
        default=(
            "checkpoints/runs/paper_b_a0.1_full/"
            "b1_cifar10_dirichlet_K5"
        ),
    )
    parser.add_argument(
        "--out", default="scripts/results/rebuttal_multiseed_paper_b.json"
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("LogicTreeNet-B replication requires a CUDA GPU")
    device = torch.device("cuda")
    properties = torch.cuda.get_device_properties(0)
    print(
        f"GPU: {torch.cuda.get_device_name(0)}, "
        f"{properties.total_memory / 2**30:.1f} GiB",
        flush=True,
    )

    train_dataset, test_dataset = get_cifar10(flatten=False)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=100, shuffle=False
    )
    probe_generator = torch.Generator().manual_seed(0)
    probe_indices = torch.randperm(
        len(train_dataset), generator=probe_generator
    )[: args.probe_size].tolist()
    probe_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(train_dataset, probe_indices),
        batch_size=100,
        shuffle=False,
    )

    rows = []
    total_start = time.time()
    for seed in args.seeds:
        seed_start = time.time()
        checkpoint_dir = _checkpoint_dir(args, seed)
        _train_missing_clients(
            args, seed, train_dataset, device, checkpoint_dir
        )
        client_accuracies, ensemble = _client_references(
            checkpoint_dir, args.num_clients, test_loader, device
        )

        collect_start = time.time()
        probe_gates, test_gates, labels, test_labels = collect_gates(
            checkpoint_dir,
            probe_loader,
            test_loader,
            device,
            num_clients=args.num_clients,
        )
        collect_sec = time.time() - collect_start
        pool_size = int(probe_gates.shape[1])

        torch.cuda.reset_peak_memory_stats()
        probe_gates = probe_gates.to(device)
        test_gates = test_gates.to(device)
        labels = labels.to(device)
        test_labels = test_labels.to(device)

        standard_start = time.time()
        bfs_accuracy, scores, _ = standard_bfs(
            probe_gates,
            test_gates,
            labels,
            test_labels,
            num_select=args.num_select,
        )
        standard_sec = time.time() - standard_start

        diverse_start = time.time()
        diverse_accuracy = diverse_bfs_streaming(
            probe_gates,
            test_gates,
            scores,
            test_labels,
            num_select=args.num_select,
            diversity_lambda=args.diversity_lambda,
        )
        diverse_sec = time.time() - diverse_start
        peak_cuda_bytes = int(torch.cuda.max_memory_allocated())
        peak_rss_kib = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        row = {
            "seed": seed,
            "checkpoint_dir": checkpoint_dir,
            "client_accuracies": [float(value) for value in client_accuracies],
            "client_mean": float(np.mean(client_accuracies)),
            "ensemble": float(ensemble),
            "bfs_full_pool": float(bfs_accuracy),
            "diverse_bfs_full_pool": float(diverse_accuracy),
            "pool_size": pool_size,
            "timing_sec": {
                "gate_collection": collect_sec,
                "standard_selection": standard_sec,
                "diverse_selection": diverse_sec,
                "seed_total": time.time() - seed_start,
            },
            "peak_cuda_bytes_selection": peak_cuda_bytes,
            "peak_rss_kib_process": peak_rss_kib,
        }
        rows.append(row)
        payload = {
            "config": vars(args),
            "raw": rows,
            "partial_summary": _summary(rows),
            "elapsed_sec": time.time() - total_start,
        }
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(
            f"seed={seed}: client={100 * row['client_mean']:.1f} "
            f"ensemble={100 * ensemble:.1f} BFS={100 * bfs_accuracy:.1f} "
            f"diverse={100 * diverse_accuracy:.1f}; "
            f"selection={(standard_sec + diverse_sec) / 60:.1f} min, "
            f"peak CUDA={peak_cuda_bytes / 2**30:.1f} GiB",
            flush=True,
        )

        del probe_gates, test_gates, labels, test_labels, scores
        torch.cuda.empty_cache()

    payload["summary"] = _summary(rows)
    payload.pop("partial_summary", None)
    with open(args.out, "w") as handle:
        json.dump(payload, handle, indent=2)
    with open(args.out.rsplit(".", 1)[0] + ".DONE", "w") as handle:
        handle.write(f"completed in {(time.time() - total_start) / 60:.1f} min\n")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
