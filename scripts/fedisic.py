"""Fed-ISIC2019 natural cross-silo experiment for the NeurIPS rebuttal.

The official six-center train/test split is read from the compact Hugging Face
parquet release. Images are center-cropped and resized to 32x32, then encoded
with the same three RGB thresholds as the paper's CIFAR experiments.

The primary metric is balanced accuracy, matching FLamby. Raw accuracy is also
reported. Local and pooled models use FLamby's class weights and focal loss,
while retaining the paper's LogicTreeNet optimizer and architecture settings.
"""

import argparse
import copy
import io
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "federated"))
from common import (  # noqa: E402
    fedavg_gate_weights,
    get_last_layer_outputs,
    make_conv_difflogic_net_tl,
    set_gate_weights,
)


FLAMBY_CLASS_WEIGHTS = torch.tensor(
    [5.5813, 2.0472, 7.0204, 26.1194, 9.5369, 101.0707, 92.5224, 38.3443]
)


class Isic32Dataset(Dataset):
    def __init__(self, images, labels, augment=False):
        self.images = images
        self.labels = labels.astype(np.int64, copy=False)
        self.targets = self.labels
        self.augment = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = torch.from_numpy(self.images[index].copy()).permute(2, 0, 1).float()
        if self.augment:
            if torch.rand(()) < 0.5:
                x = x.flip(2)
            if torch.rand(()) < 0.5:
                x = x.flip(1)
            x = torch.rot90(x, int(torch.randint(0, 4, ()).item()), (1, 2))
            contrast = 0.85 + 0.30 * torch.rand(())
            brightness = torch.randint(-12, 13, ()).float()
            x = (x * contrast + brightness).clamp_(0, 255)
        x = x / 255.0
        x = torch.cat([(x > threshold).float() for threshold in (0.25, 0.5, 0.75)])
        return x, int(self.labels[index])


def prepare_cache(data_dir, cache_path):
    import pyarrow.parquet as pq

    arrays = {}
    for split in ("train", "test"):
        path = Path(data_dir) / f"{split}.parquet"
        table = pq.read_table(path, columns=["image", "center", "label"])
        rows = table.to_pylist()
        images = np.empty((len(rows), 32, 32, 3), dtype=np.uint8)
        centers = np.empty(len(rows), dtype=np.uint8)
        labels = np.empty(len(rows), dtype=np.uint8)
        for i, row in enumerate(rows):
            image = Image.open(io.BytesIO(row["image"]["bytes"])).convert("RGB")
            width, height = image.size
            side = min(width, height)
            left = (width - side) // 2
            top = (height - side) // 2
            image = image.crop((left, top, left + side, top + side))
            image = image.resize((32, 32), Image.Resampling.BILINEAR)
            images[i] = np.asarray(image, dtype=np.uint8)
            centers[i] = row["center"]
            labels[i] = row["label"]
            if (i + 1) % 2000 == 0:
                print(f"  cached {split}: {i + 1}/{len(rows)}", flush=True)
        arrays[f"{split}_images"] = images
        arrays[f"{split}_centers"] = centers
        arrays[f"{split}_labels"] = labels

    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, **arrays)
    print(f"Wrote {cache_path}", flush=True)


def balanced_accuracy(labels, predictions, num_classes=8):
    recalls = []
    for cls in range(num_classes):
        mask = labels == cls
        if mask.any():
            recalls.append(float((predictions[mask] == cls).mean()))
    return float(np.mean(recalls))


def metrics_from_predictions(labels, predictions, centers):
    result = {
        "accuracy": float((predictions == labels).mean()),
        "balanced_accuracy": balanced_accuracy(labels, predictions),
    }
    result["per_center"] = {}
    for center in sorted(np.unique(centers)):
        mask = centers == center
        result["per_center"][str(int(center))] = {
            "n": int(mask.sum()),
            "accuracy": float((predictions[mask] == labels[mask]).mean()),
            "balanced_accuracy": balanced_accuracy(labels[mask], predictions[mask]),
        }
    return result


def focal_loss(logits, targets, class_weights, gamma=2.0):
    log_probs = F.log_softmax(logits, dim=1)
    log_pt = log_probs.gather(1, targets[:, None]).squeeze(1)
    pt = log_pt.exp()
    weights = class_weights.to(logits.device)[targets]
    return (-(1.0 - pt).pow(gamma) * weights * log_pt).mean()


def train_model(model, loader, num_iters, lr, device, loss_kind, weight_decay):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    iterator = iter(loader)
    model.train()
    started = time.time()
    for iteration in range(1, num_iters + 1):
        try:
            x, y = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            x, y = next(iterator)
        x, y = x.to(device), y.to(device)
        logits = model(x)
        if loss_kind == "focal":
            loss = focal_loss(logits, y, FLAMBY_CLASS_WEIGHTS)
        else:
            loss = F.cross_entropy(logits, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if iteration % max(1, num_iters // 5) == 0:
            print(
                f"    iter {iteration}/{num_iters}, loss={loss.item():.4f}, "
                f"elapsed={(time.time() - started) / 60:.1f}min",
                flush=True,
            )
    return time.time() - started


def predict_logits(model, loader, device):
    model.to(device)
    model.eval()
    outputs = []
    with torch.no_grad():
        for x, _ in loader:
            outputs.append(model(x.to(device)).detach().cpu())
    model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return torch.cat(outputs).numpy()


def weighted_average(client_models, sample_counts):
    weights = torch.tensor(sample_counts, dtype=torch.float64)
    weights /= weights.sum()
    merged = {}
    for name, _ in client_models[0].named_parameters():
        tensors = [dict(model.named_parameters())[name].data for model in client_models]
        merged[name] = sum(float(weight) * tensor for weight, tensor in zip(weights, tensors))
    return merged


def collect_gate_pool(models, loader, device):
    matrices = []
    labels = None
    for index, model in enumerate(models):
        model.to(device)
        gates, labels = get_last_layer_outputs(
            model, loader, device, binarize=False, flat=False
        )
        model.cpu()
        matrices.append((gates > 0.5).to(torch.uint8))
        print(f"  collected gate outputs from model {index}", flush=True)
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return matrices, labels.numpy()


def select_basic(probe_matrices, labels, num_select, num_classes=8):
    pool = torch.cat(probe_matrices, dim=1)
    scores = []
    for cls in range(num_classes):
        in_class = pool[labels == cls].float().mean(0)
        out_class = pool[labels != cls].float().mean(0)
        scores.append(in_class - out_class)
    scores = torch.stack(scores)
    selected = {
        cls: scores[cls].argsort(descending=True)[:num_select]
        for cls in range(num_classes)
    }
    return selected, scores


def select_diverse(
    probe_matrices,
    scores,
    num_select,
    diversity_lambda,
    preselect,
    num_classes=8,
):
    pool = torch.cat(probe_matrices, dim=1)
    selected = {}
    for cls in range(num_classes):
        candidates = scores[cls].argsort(descending=True)[:preselect]
        values = pool[:, candidates].float()
        values -= values.mean(0, keepdim=True)
        values /= values.norm(dim=0, keepdim=True).clamp_min_(1e-8)
        max_correlation = torch.zeros(len(candidates))
        chosen_positions = []
        for _ in range(num_select):
            adjusted = scores[cls, candidates] - diversity_lambda * max_correlation
            if chosen_positions:
                adjusted[torch.tensor(chosen_positions)] = -float("inf")
            best = int(adjusted.argmax())
            chosen_positions.append(best)
            correlation = torch.mv(values.T, values[:, best]).abs()
            max_correlation = torch.maximum(max_correlation, correlation)
        selected[cls] = candidates[torch.tensor(chosen_positions)]
        print(f"  diverse selection class {cls + 1}/{num_classes}", flush=True)
    return selected


def predict_selected(test_matrices, selected, num_classes=8):
    num_gates = test_matrices[0].shape[1]
    logits = torch.zeros(test_matrices[0].shape[0], num_classes)
    for cls in range(num_classes):
        global_indices = selected[cls]
        for model_index, matrix in enumerate(test_matrices):
            mask = global_indices.div(num_gates, rounding_mode="floor") == model_index
            local_indices = global_indices[mask] % num_gates
            if len(local_indices):
                logits[:, cls] += matrix[:, local_indices].sum(1).float()
    return logits.argmax(1).numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data-fedisic")
    parser.add_argument("--cache", default="data-fedisic/fedisic32.npz")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--size", choices=["paper_s", "paper_b"], default="paper_s")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-iters", type=int, default=10000)
    parser.add_argument("--central-iters", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--weight-decay", type=float, default=0.002)
    parser.add_argument("--loss", choices=["focal", "ce"], default="focal")
    parser.add_argument("--probe-size", type=int, default=2400)
    parser.add_argument("--num-select", type=int, default=100)
    parser.add_argument("--diversity-lambda", type=float, default=0.3)
    parser.add_argument("--diverse-preselect", type=int, default=2000)
    parser.add_argument("--skip-diverse", action="store_true")
    parser.add_argument("--skip-central", action="store_true")
    parser.add_argument("--checkpoint-dir", default="scripts/checkpoints/fedisic")
    parser.add_argument("--out", default="scripts/results/rebuttal_natural_fedisic.json")
    args = parser.parse_args()

    if not os.path.exists(args.cache):
        prepare_cache(args.data_dir, args.cache)
    if args.prepare_only:
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("Paper-S/B Fed-ISIC training requires CUDA")
    print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    data = np.load(args.cache)
    train_images = data["train_images"]
    train_labels = data["train_labels"].astype(np.int64)
    train_centers = data["train_centers"]
    test_labels = data["test_labels"].astype(np.int64)
    test_centers = data["test_centers"]
    train_ds = Isic32Dataset(train_images, train_labels, augment=True)
    train_eval_ds = Isic32Dataset(train_images, train_labels, augment=False)
    test_ds = Isic32Dataset(data["test_images"], test_labels, augment=False)
    test_loader = DataLoader(test_ds, batch_size=200, shuffle=False, num_workers=2)

    client_indices = [np.flatnonzero(train_centers == center).tolist() for center in range(6)]
    sample_counts = [len(indices) for indices in client_indices]
    print(f"Client sizes: {sample_counts}", flush=True)

    run_started = time.time()
    result = {
        "config": vars(args),
        "data": {
            "train_size": len(train_labels),
            "test_size": len(test_labels),
            "client_sizes": sample_counts,
            "train_class_counts": np.bincount(train_labels, minlength=8).tolist(),
            "test_class_counts": np.bincount(test_labels, minlength=8).tolist(),
        },
        "methods": {},
        "training_seconds": {},
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = (
        Path(args.checkpoint_dir)
        / f"{args.size}_{args.loss}_i{args.num_iters}_seed{args.seed}"
    )
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    base_model = make_conv_difflogic_net_tl(
        classes=8, depth=2, size=args.size, device=device.type
    ).cpu()
    client_models = []
    for center, indices in enumerate(client_indices):
        model = copy.deepcopy(base_model)
        checkpoint = checkpoint_dir / f"client_{center}.pt"
        if checkpoint.exists():
            model.load_state_dict(torch.load(checkpoint, map_location="cpu")["state_dict"])
            elapsed = 0.0
            print(f"Loaded client {center}: {checkpoint}", flush=True)
        else:
            loader = DataLoader(
                Subset(train_ds, indices),
                batch_size=min(args.batch_size, len(indices)),
                shuffle=True,
                num_workers=2,
                pin_memory=True,
            )
            print(f"Training client {center} ({len(indices)} images)", flush=True)
            model.to(device)
            elapsed = train_model(
                model,
                loader,
                args.num_iters,
                args.lr,
                device,
                args.loss,
                args.weight_decay,
            )
            model.cpu()
            torch.save({"state_dict": model.state_dict(), "config": vars(args)}, checkpoint)
            torch.cuda.empty_cache()
        result["training_seconds"][f"client_{center}"] = elapsed
        client_models.append(model)
        with open(args.out, "w") as handle:
            json.dump(result, handle, indent=2)

    client_logits = []
    for center, model in enumerate(client_models):
        logits = predict_logits(model, test_loader, device)
        client_logits.append(logits)
        result["methods"][f"client_{center}"] = metrics_from_predictions(
            test_labels, logits.argmax(1), test_centers
        )
    client_bacc = [
        result["methods"][f"client_{center}"]["balanced_accuracy"] for center in range(6)
    ]
    result["methods"]["client_mean"] = {
        "balanced_accuracy": float(np.mean(client_bacc)),
        "balanced_accuracy_std_across_clients": float(np.std(client_bacc)),
    }

    ensemble_predictions = np.mean(client_logits, axis=0).argmax(1)
    result["methods"]["ensemble"] = metrics_from_predictions(
        test_labels, ensemble_predictions, test_centers
    )

    for name, weights in (
        ("fedavg_uniform", fedavg_gate_weights(client_models)),
        ("fedavg_sample_weighted", weighted_average(client_models, sample_counts)),
    ):
        merged = copy.deepcopy(base_model)
        set_gate_weights(merged, weights)
        predictions = predict_logits(merged, test_loader, device).argmax(1)
        result["methods"][name] = metrics_from_predictions(
            test_labels, predictions, test_centers
        )
        del merged

    if not args.skip_central:
        central = copy.deepcopy(base_model)
        central_checkpoint = checkpoint_dir / "central.pt"
        if central_checkpoint.exists():
            central.load_state_dict(
                torch.load(central_checkpoint, map_location="cpu")["state_dict"]
            )
            elapsed = 0.0
            print(f"Loaded pooled model: {central_checkpoint}", flush=True)
        else:
            central_loader = DataLoader(
                train_ds,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=True,
            )
            print("Training pooled reference", flush=True)
            central.to(device)
            elapsed = train_model(
                central,
                central_loader,
                args.central_iters or args.num_iters,
                args.lr,
                device,
                args.loss,
                args.weight_decay,
            )
            central.cpu()
            torch.save(
                {"state_dict": central.state_dict(), "config": vars(args)},
                central_checkpoint,
            )
            torch.cuda.empty_cache()
        result["training_seconds"]["central"] = elapsed
        predictions = predict_logits(central, test_loader, device).argmax(1)
        result["methods"]["pooled"] = metrics_from_predictions(
            test_labels, predictions, test_centers
        )
        del central

    generator = torch.Generator().manual_seed(args.seed + 1000)
    probe_indices = torch.randperm(len(train_eval_ds), generator=generator)[
        : args.probe_size
    ].tolist()
    probe_loader = DataLoader(
        Subset(train_eval_ds, probe_indices),
        batch_size=200,
        shuffle=False,
        num_workers=2,
    )
    print("Collecting FBFS probe gates", flush=True)
    probe_matrices, probe_labels = collect_gate_pool(client_models, probe_loader, device)
    test_matrices, gate_test_labels = collect_gate_pool(client_models, test_loader, device)
    assert np.array_equal(gate_test_labels, test_labels)

    basic_selected, scores = select_basic(
        probe_matrices, probe_labels, args.num_select
    )
    predictions = predict_selected(test_matrices, basic_selected)
    result["methods"]["fbfs_basic"] = metrics_from_predictions(
        test_labels, predictions, test_centers
    )

    if not args.skip_diverse:
        diverse_selected = select_diverse(
            probe_matrices,
            scores,
            args.num_select,
            args.diversity_lambda,
            min(args.diverse_preselect, scores.shape[1]),
        )
        predictions = predict_selected(test_matrices, diverse_selected)
        result["methods"]["fbfs_diverse"] = metrics_from_predictions(
            test_labels, predictions, test_centers
        )

    result["elapsed_seconds"] = time.time() - run_started
    with open(args.out, "w") as handle:
        json.dump(result, handle, indent=2)
    Path(args.out).with_suffix(".DONE").write_text(
        f"{result['elapsed_seconds'] / 60:.1f}min\n"
    )

    print("\nBalanced accuracy / accuracy", flush=True)
    for name, values in result["methods"].items():
        if "accuracy" in values:
            print(
                f"{name:24s} {100 * values['balanced_accuracy']:6.2f} / "
                f"{100 * values['accuracy']:6.2f}",
                flush=True,
            )
        else:
            print(f"{name:24s} {100 * values['balanced_accuracy']:6.2f}", flush=True)
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
