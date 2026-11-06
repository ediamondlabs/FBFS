"""Shared utilities for federated difflogic experiments (Approach B)."""

import math
import os
import pickle
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms

# Optional: torchlogix for GPU-accelerated difflogic
try:
    from torchlogix.layers import (
        LogicDense as TLLogicDense,
        LogicConv2d as TLLogicConv2d,
        OrPooling2d as TLOrPooling2d,
        GroupSum as TLGroupSum,
    )
    HAS_TORCHLOGIX = True
except ImportError:
    HAS_TORCHLOGIX = False


# -- Data utilities ------------------------------------------------------------

def get_mnist(data_dir='./data'):
    tf = transforms.Compose([transforms.ToTensor()])
    train = datasets.MNIST(data_dir, train=True, download=True, transform=tf)
    test = datasets.MNIST(data_dir, train=False, download=True, transform=tf)
    return train, test


def get_fashion_mnist(data_dir='./data-fashion'):
    tf = transforms.Compose([transforms.ToTensor()])
    train = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tf)
    test = datasets.FashionMNIST(data_dir, train=False, download=True, transform=tf)
    return train, test


def get_cifar10(data_dir='./data-cifar', num_thresholds=3, flatten=True):
    tf_list = [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat(
            [(x > (i+1) / (num_thresholds+1)).float() for i in range(num_thresholds)], dim=0
        )),
    ]
    if flatten:
        tf_list.append(transforms.Lambda(lambda x: x.view(-1)))  # flatten to 1D
    transform = transforms.Compose(tf_list)
    train = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
    test = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
    return train, test


def get_cifar100(data_dir='./data-cifar100', num_thresholds=3, flatten=True):
    tf_list = [
        transforms.ToTensor(),
        transforms.Lambda(lambda x: torch.cat(
            [(x > (i+1) / (num_thresholds+1)).float() for i in range(num_thresholds)], dim=0
        )),
    ]
    if flatten:
        tf_list.append(transforms.Lambda(lambda x: x.view(-1)))
    transform = transforms.Compose(tf_list)
    # Bypass md5 integrity checks to allow loading from non-toronto.edu mirrors
    # (parquet from HuggingFace converted to pickle locally).
    datasets.CIFAR100._check_integrity = lambda self: True
    orig_load_meta = datasets.CIFAR100._load_meta
    def _patched_load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        with open(path, 'rb') as f:
            data = pickle.load(f, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}
    datasets.CIFAR100._load_meta = _patched_load_meta
    train = datasets.CIFAR100(data_dir, train=True, download=False, transform=transform)
    test = datasets.CIFAR100(data_dir, train=False, download=False, transform=transform)
    return train, test


def partition_iid(dataset, K):
    n = len(dataset) // K
    idx = list(range(len(dataset)))
    random.shuffle(idx)
    return [idx[i*n:(i+1)*n] for i in range(K)]


def partition_noniid(dataset, K, classes_per_client=2):
    targets = np.array(dataset.targets)
    class_idx = {c: np.where(targets == c)[0].tolist() for c in range(10)}
    for c in class_idx:
        random.shuffle(class_idx[c])

    assignments = []
    for i in range(K):
        assignments.append([(i * classes_per_client + j) % 10 for j in range(classes_per_client)])

    class_clients = {c: [] for c in range(10)}
    for cid, classes in enumerate(assignments):
        for c in classes:
            class_clients[c].append(cid)

    client_idx = [[] for _ in range(K)]
    for c in range(10):
        clients = class_clients[c]
        idx = class_idx[c]
        n = len(idx) // len(clients)
        for i, cid in enumerate(clients):
            client_idx[cid].extend(idx[i*n:(i+1)*n])

    for i, (idx, _cls) in enumerate(zip(client_idx, assignments)):
        actual = sorted(set(targets[idx].tolist()))
        print(f"  Client {i}: {len(idx)} samples, classes {actual}")
    return client_idx


def partition_dirichlet(dataset, K, alpha=0.5, num_classes=10):
    """Dirichlet non-IID partition. alpha→0 = extreme non-IID, alpha→∞ = IID."""
    targets = np.array(dataset.targets)
    client_idx = [[] for _ in range(K)]
    for c in range(num_classes):
        idx_c = np.where(targets == c)[0]
        np.random.shuffle(idx_c)
        proportions = np.random.dirichlet([alpha] * K)
        # Convert proportions to actual counts
        counts = (proportions * len(idx_c)).astype(int)
        counts[-1] = len(idx_c) - counts[:-1].sum()  # fix rounding
        start = 0
        for k in range(K):
            client_idx[k].extend(idx_c[start:start + counts[k]].tolist())
            start += counts[k]
    # Print distribution
    for i in range(K):
        classes = sorted(set(targets[client_idx[i]].tolist()))
        counts_per_class = {c: (targets[client_idx[i]] == c).sum() for c in classes}
        print(f"  Client {i}: {len(client_idx[i])} samples, classes {classes}")
    return client_idx


# -- Difflogic layers ---------------------------------------------------------

def bin_op_all(a, b):
    """All 16 binary gate outputs at once. Returns (..., num_gates, 16)."""
    ab = a * b
    a_plus_b = a + b
    gates = torch.stack([
        torch.zeros_like(a),       # 0:  FALSE
        ab,                         # 1:  AND
        a - ab,                     # 2:  A AND NOT B
        a,                          # 3:  A
        b - ab,                     # 4:  B AND NOT A
        b,                          # 5:  B
        a_plus_b - 2 * ab,          # 6:  XOR
        a_plus_b - ab,              # 7:  OR
        1 - (a_plus_b - ab),        # 8:  NOR
        1 - (a_plus_b - 2 * ab),    # 9:  XNOR
        1 - b,                      # 10: NOT B
        1 - b + ab,                 # 11: B IMPLIES A
        1 - a,                      # 12: NOT A
        1 - a + ab,                 # 13: A IMPLIES B
        1 - ab,                     # 14: NAND
        torch.ones_like(a),         # 15: TRUE
    ], dim=-1)
    return gates


def bin_op_weighted(a, b, w):
    """Compute weighted combination of all 16 gate outputs directly.

    Any weighted sum of the 16 boolean gates decomposes as:
        out = c0 + c_a * a + c_b * b + c_ab * a * b
    where c0, c_a, c_b, c_ab are linear combinations of the 16 weights.

    This avoids materializing the full (... , 16) intermediate tensor,
    reducing memory by ~4x and improving speed significantly for conv layers.
    """
    c0 = (w[..., 8] + w[..., 9] + w[..., 10] + w[..., 11]
          + w[..., 12] + w[..., 13] + w[..., 14] + w[..., 15])
    c_a = (w[..., 2] + w[..., 3] + w[..., 6] + w[..., 7]
           - w[..., 8] - w[..., 9] - w[..., 12] - w[..., 13])
    c_b = (w[..., 4] + w[..., 5] + w[..., 6] + w[..., 7]
           - w[..., 8] - w[..., 9] - w[..., 10] - w[..., 11])
    c_ab = (w[..., 1] - w[..., 2] - w[..., 4] - 2 * w[..., 6] - w[..., 7]
            + w[..., 8] + 2 * w[..., 9] + w[..., 11] + w[..., 13] - w[..., 14])
    return c0 + c_a * a + c_b * b + c_ab * a * b


class LogicTreeConv2d(nn.Module):
    """Logic gate tree convolution (Petersen et al., NeurIPS 2024).

    Each output channel is a complete binary tree of depth `depth`.
    Leaves sample from a spatial receptive field (kernel_size x kernel_size)
    across at most 2 input channels. Gates use the same 16-op softmax as
    LogicLayer. The tree is applied convolutionally (shared weights across
    positions) via F.unfold.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, depth=2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.depth = depth
        self.n_leaves = 2 ** depth
        self.n_gates = 2 ** depth - 1

        # Gate logits: (out_channels, n_gates, 16)
        weights = torch.zeros(out_channels, self.n_gates, 16)
        # Residual init: gate type 3 (pass-through A) gets high logit
        weights[:, :, 3] = 5.0
        self.weights = nn.Parameter(weights)

        # Connection indices for leaves: which (channel, h_offset, w_offset)
        # Each tree reads from at most 2 input channels
        conn_c = torch.zeros(out_channels, self.n_leaves, dtype=torch.long)
        conn_hw = torch.zeros(out_channels, self.n_leaves, dtype=torch.long)

        for oc in range(out_channels):
            # Pick 2 random input channels for this tree
            if in_channels >= 2:
                ch_pair = torch.randperm(in_channels)[:2]
            else:
                ch_pair = torch.zeros(2, dtype=torch.long)
            # Assign each leaf to one of the 2 channels
            conn_c[oc] = ch_pair[torch.randint(0, 2, (self.n_leaves,))]
            # Assign each leaf a random spatial offset within the kernel
            conn_hw[oc] = torch.randint(0, kernel_size * kernel_size, (self.n_leaves,))

        self.register_buffer('conn_c', conn_c)
        self.register_buffer('conn_hw', conn_hw)

    def forward(self, x):
        B, C, H, W = x.shape
        k = self.kernel_size
        H_out = H + 2 * self.padding - k + 1
        W_out = W + 2 * self.padding - k + 1
        L = H_out * W_out

        # Extract patches via pad+slice (MPS-compatible, avoids F.unfold/im2col)
        x_pad = F.pad(x, [self.padding] * 4)
        slices = []
        for di in range(k):
            for dj in range(k):
                slices.append(x_pad[:, :, di:di + H_out, dj:dj + W_out])
        # patches: (B, C, kH*kW, H_out, W_out) -> (B, C, kH*kW, L)
        patches = torch.stack(slices, dim=2).reshape(B, C, k * k, L)

        # Gather leaf values: (B, out_c, n_leaves, L)
        leaf_vals = patches[:, self.conn_c, self.conn_hw, :]

        # Evaluate tree bottom-up
        # Level 0: pair leaves -> n_leaves/2 outputs using gates [0 .. n_leaves/2 - 1]
        # Level 1: pair level-0 outputs -> using gates [n_leaves/2 .. 3*n_leaves/4 - 1]
        # etc.
        current = leaf_vals  # (B, out_c, n_leaves, L)
        gate_idx = 0
        for level in range(self.depth):
            n_pairs = current.shape[2] // 2
            a = current[:, :, 0::2, :]  # (B, out_c, n_pairs, L)
            b = current[:, :, 1::2, :]  # (B, out_c, n_pairs, L)

            # Get gate weights for this level
            w = self.weights[:, gate_idx:gate_idx + n_pairs, :]  # (out_c, n_pairs, 16)
            if self.training:
                w = F.softmax(w, dim=-1)
            else:
                w = F.one_hot(w.argmax(-1), 16).float()

            # Compute weighted gate output directly (avoids 16x memory blowup)
            # w: (out_c, n_pairs, 16) -> (1, out_c, n_pairs, 1, 16)
            w_bcast = w.unsqueeze(0).unsqueeze(3)
            # a, b: (B, out_c, n_pairs, L)
            current = bin_op_weighted(a, b, w_bcast)

            gate_idx += n_pairs

        # current: (B, out_c, 1, L) -> squeeze to (B, out_c, L)
        output = current.squeeze(2)
        # Reshape to (B, out_c, H_out, W_out)
        return output.view(B, self.out_channels, H_out, W_out)


class ORPool2d(nn.Module):
    """OR pooling via max (the t-conorm relaxation of OR)."""

    def __init__(self, kernel_size=2, stride=2):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride)

    def forward(self, x):
        return self.pool(x)


class LogicLayer(nn.Module):
    def __init__(self, in_dim, out_dim, gumbel=False, gumbel_tau=1.0):
        super().__init__()
        self.weights = nn.Parameter(torch.randn(out_dim, 16))
        self.gumbel = gumbel
        self.gumbel_tau = gumbel_tau
        assert out_dim * 2 >= in_dim
        c = torch.randperm(in_dim)[torch.randperm(2 * out_dim) % in_dim].reshape(2, out_dim)
        self.register_buffer('idx_a', c[0].long())
        self.register_buffer('idx_b', c[1].long())

    def forward(self, x):
        a, b = x[..., self.idx_a], x[..., self.idx_b]
        if self.training:
            if self.gumbel:
                # Gumbel-Softmax straight-through: hard forward, soft backward
                w_soft = F.gumbel_softmax(self.weights, tau=self.gumbel_tau, hard=True)
            else:
                w_soft = F.softmax(self.weights, dim=-1)
            w = w_soft
        else:
            w = F.one_hot(self.weights.argmax(-1), 16).float()
        return (bin_op_all(a, b) * w).sum(dim=-1)


class GroupSum(nn.Module):
    def __init__(self, k, tau=1.0):
        super().__init__()
        self.k, self.tau = k, tau

    def forward(self, x):
        return x.reshape(*x.shape[:-1], self.k, x.shape[-1] // self.k).sum(-1) / self.tau


def make_difflogic_net(in_dim, neurons, layers, classes, tau=10.0, gumbel=False):
    mods = [nn.Flatten(), LogicLayer(in_dim, neurons, gumbel=gumbel)]
    for _ in range(layers - 1):
        mods.append(LogicLayer(neurons, neurons, gumbel=gumbel))
    mods.append(GroupSum(classes, tau=tau))
    return nn.Sequential(*mods)


def make_conv_difflogic_net(classes=10, tau=10.0, depth=2, size='small'):
    """Conv difflogic for CIFAR-10 (9-channel input).
    small: 2.7K gates / 43K params (Mac MPS).  medium: 9.2K / 148K.
    large: 57K / 912K (~paper-S, needs GPU; 6 paired convs)."""
    cv = lambda ci, co: LogicTreeConv2d(ci, co, kernel_size=3, padding=1, depth=depth)
    cfg = {'small':  ([(9,32),(32,64),(64,128)],   2048, 1024, 1000),
           'medium': ([(9,128),(128,128),(128,128)], 2048, 4096, 4000),
           'large':  ([(9,128),(128,128),(128,128),(128,128),(128,128),(128,128)],
                      2048, 8192, 8000)}[size]
    convs, h0, h1, h2 = cfg
    paired = (size == 'large')  # large: 2 convs per pool block
    body = []
    for i, (ci, co) in enumerate(convs):
        body.append(cv(ci, co))
        if (not paired and True) or (paired and i % 2 == 1):
            body.append(ORPool2d(2))
    return nn.Sequential(*body, nn.Flatten(),
                         LogicLayer(h0, h1), LogicLayer(h1, h2),
                         GroupSum(classes, tau=tau))


def make_conv_difflogic_net_tl(classes=10, tau=10.0, depth=2, size='small', device='cpu'):
    """Same architectures as make_conv_difflogic_net but with torchlogix (GPU)."""
    if not HAS_TORCHLOGIX:
        raise ImportError("torchlogix not installed. pip install torchlogix")
    # cfg[size] = (channels-progression, fc-progression, tree_depth, tau, paired_convs, parametrization)
    cfg = {
        'paper_b': ([9, 512, 2048, 8192, 16384],         [65536, 1310720, 655360, 327680], 3,    280.0, False, 'light'),
        'paper_s': ([9, 32,  128,  512,  1024],          [4096,  40960,   20480,  10240],  3,    20.0,  False, None),
        # Same conv stack as paper_s but final FC sized for CIFAR-100 (must
        # divide by 100 for GroupSum); 10000 / 100 = 100 gates per group.
        'paper_s_c100': ([9, 32, 128, 512, 1024],         [4096,  40000,   20000,  10000],  3,    20.0,  False, None),
        # Wider final FC: 100000 / 100 = 1000 gates per class, matching the
        # per-class capacity of paper_s on CIFAR-10 (10240 / 10 = 1024).
        'paper_s_c100_xl': ([9, 32, 128, 512, 1024],      [4096,  40000,   40000, 100000],  3,    20.0,  False, None),
        'large':   ([9, 128, 128, 128, 128, 128, 128],   [2048, 8192,    8000],            depth, tau,   True,  None),
        'medium':  ([9, 128, 128, 128],                   [2048, 4096,    4000],            depth, tau,   False, None),
        'small':   ([9, 32,  64,  128],                   [2048, 1024,    1000],            depth, tau,   False, None),
    }[size]
    chs, fcs, td, t, paired, p = cfg
    p_kw = {'parametrization': p} if p else {}
    pool = lambda: TLOrPooling2d(kernel_size=2, stride=2)
    spatial = [32 >> (i // (2 if paired else 1)) for i in range(len(chs)-1)]
    body = []
    for i, (sd, ci, co) in enumerate(zip(spatial, chs[:-1], chs[1:])):
        body.append(TLLogicConv2d(in_dim=sd, channels=ci, num_kernels=co, tree_depth=td,
                                  receptive_field_size=3, padding=1, device=device, **p_kw))
        if not paired or i % 2 == 1:
            body.append(pool())
    fc_layers = [TLLogicDense(fcs[i], fcs[i+1], device=device, **p_kw) for i in range(len(fcs)-1)]
    return nn.Sequential(*body, nn.Flatten(), *fc_layers, TLGroupSum(k=classes, tau=t, device=device))


# -- Evaluation ----------------------------------------------------------------

def evaluate(model, loader, device, binarize=True, hard=False, flat=True):
    was_training = model.training
    model.eval() if hard else model.train()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if binarize:
                x = (x > 0.5).float()
            if flat:
                x = x.view(x.size(0), -1)
            correct += (model(x).argmax(1) == y).sum().item()
            total += x.size(0)
    model.train(was_training)
    return correct / total


def _eval_loop(reduce_fn, models, loader, device, binarize, hard, flat):
    """Shared eval loop. `reduce_fn(stack_of_logits)` -> (B, C) merged logits."""
    for m in models:
        m.eval() if hard else m.train()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            if binarize: x = (x > 0.5).float()
            if flat:     x = x.view(x.size(0), -1)
            logits = reduce_fn(torch.stack([m(x) for m in models]))
            correct += (logits.argmax(1) == y).sum().item()
            total   += x.size(0)
    return correct / total


def evaluate_ensemble(models, loader, device, binarize=True, hard=False, flat=True):
    return _eval_loop(lambda s: s.mean(0), models, loader, device, binarize, hard, flat)


def evaluate_ensemble_weighted(models, loader, device, binarize=True, hard=False, flat=True):
    """Confidence-weighted ensemble: per-input, weight each client by negative entropy."""
    def reduce(s):                                # s: (K, B, C)
        p = F.softmax(s, dim=-1)
        h = -(p * torch.log(p + 1e-10)).sum(-1)   # (K, B)
        w = F.softmax(-h, dim=0).unsqueeze(-1)    # (K, B, 1)
        return (s * w).sum(0)
    return _eval_loop(reduce, models, loader, device, binarize, hard, flat)


# -- Client training helpers --------------------------------------------------

def train_one_client(model, loader, num_iters, lr=0.01, device='cpu',
                     binarize=True, flat=True, optimizer_kind='adam',
                     weight_decay=0.0):
    """Train one client model for `num_iters` iterations of CE on `loader`.

    Bit-identical to the inline loops in trilemma_run, trilemma_high_r,
    bfs_budget_sweep, probe_robustness, verify_bfs_federated, distill_tradeoff
    (without their eval prints).
    """
    import torch.optim as optim
    if optimizer_kind == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr)
    gen = data_iter(loader)
    model.train()
    for _ in range(num_iters):
        x, y = next(gen)
        x, y = x.to(device), y.to(device)
        x_in = x.view(x.size(0), -1) if flat else x
        if binarize:
            x_in = (x_in > 0.5).float()
        loss = F.cross_entropy(model(x_in), y)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
    return model


def train_k_clients(global_model, train_ds, num_clients, alpha, num_iters,
                    lr=0.01, batch_size=128, device='cpu', binarize=True,
                    flat=True, optimizer_kind='adam', weight_decay=0.0,
                    seed=42, client_idx=None):
    """Partition train_ds via Dirichlet(alpha), train K client copies of `global_model`.

    Returns (client_models, client_loaders, client_idx). Caller may pass an
    existing `client_idx` to skip partitioning and reuse it.
    """
    import copy
    if client_idx is None:
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
        client_idx = partition_dirichlet(train_ds, num_clients, alpha=alpha)
    client_loaders = [
        torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_ds, idx),
            batch_size=batch_size, shuffle=True, drop_last=True)
        for idx in client_idx]
    client_models = [copy.deepcopy(global_model) for _ in range(num_clients)]
    for m, L in zip(client_models, client_loaders):
        train_one_client(m, L, num_iters, lr=lr, device=device, binarize=binarize,
                         flat=flat, optimizer_kind=optimizer_kind,
                         weight_decay=weight_decay)
    return client_models, client_loaders, client_idx


# -- Federated helpers ---------------------------------------------------------

def get_gate_weights(model):
    """Extract gate logit tensors (trainable parameters)."""
    return {n: p.data.clone() for n, p in model.named_parameters()}


def set_gate_weights(model, weights):
    """Load gate logit tensors into a model (wiring buffers unchanged)."""
    for n, p in model.named_parameters():
        p.data.copy_(weights[n])


def _stack_param(client_models, name):
    """Stack one named parameter across clients into a (K, ...) tensor."""
    return torch.stack([dict(m.named_parameters())[name].data for m in client_models])


def _is_gate(t):
    """Heuristic for "this is a gate-logit tensor" (last dim 4 or 16, ndim>=3)."""
    return t.dim() >= 3 and t.shape[-1] in (4, 16)


def _merge_each(client_models, gate_fn):
    """Apply gate_fn(stacked) to gate tensors, fall back to mean for non-gates."""
    return {n: gate_fn(s) if _is_gate(s := _stack_param(client_models, n)) else s.mean(0)
            for n, _ in client_models[0].named_parameters()}


def fedavg_gate_weights(client_models):
    """Average gate logit tensors across client models."""
    return _merge_each(client_models, lambda s: s.mean(0))


def fedavg_gate_weights_entropy(client_models):
    """Entropy-weighted FedAvg: per-gate, weight by inverse entropy."""
    def merge(s):
        p = F.softmax(s, dim=-1)
        h = -(p * torch.log(p + 1e-10)).sum(-1)
        w = F.softmax(-h, dim=0)
        return (s * w.unsqueeze(-1)).sum(0)
    return _merge_each(client_models, merge)


def fedavg_gate_voting(client_models):
    """Discrete gate voting: majority-vote on argmax type; tie-break by max confidence.

    Result logits set to 10.0 at winner, 0 elsewhere. Data-free.
    """
    def merge(s):
        K, *gshape, D = s.shape
        flat = s.reshape(K, -1, D)                      # (K, G, D)
        votes = flat.argmax(-1)                         # (K, G)
        conf = flat.max(-1).values                      # (K, G)
        # One-hot vote tally: (G, D)
        counts = F.one_hot(votes, D).sum(0)
        # Tie-break: among top-count types, pick one whose most-confident
        # supporter has the highest logit. We score each (g, d) by
        # (count==max) * (max_k conf[k,g] | votes[k,g]==d), then argmax over d.
        max_count = counts.max(-1, keepdim=True).values
        is_top = (counts == max_count)                  # (G, D)
        # For each (g, d): max conf among voters of d (or -inf if none).
        K_, G = votes.shape
        # (K, G, D): per-client one-hot of its vote, scaled by its conf.
        conf_oh = F.one_hot(votes, D).float() * conf.unsqueeze(-1)
        # Replace zeros (non-voters) with -inf so max ignores them.
        conf_oh = torch.where(F.one_hot(votes, D).bool(), conf_oh,
                              torch.full_like(conf_oh, float('-inf')))
        best_conf_per_d = conf_oh.max(0).values         # (G, D)
        # Mask out non-top types, then argmax to pick the winner.
        score = torch.where(is_top, best_conf_per_d,
                            torch.full_like(best_conf_per_d, float('-inf')))
        winners = score.argmax(-1)                      # (G,)
        out = (F.one_hot(winners, D).float() * 10.0).reshape(*gshape, D)
        return out
    return _merge_each(client_models, merge)


def fedavg_gate_ties(client_models, init_weights=None):
    """TIES-style: agree on argmax → uniform mean; disagree → most-confident client."""
    stats = {'agree': 0, 'disagree': 0}
    def merge(s):
        K, *gshape, D = s.shape
        flat = s.reshape(K, -1, D)                      # (K, G, D)
        votes = flat.argmax(-1)                         # (K, G)
        conf = flat.max(-1).values                      # (K, G)
        agree = (votes == votes[0:1]).all(0)            # (G,)
        best_k = conf.argmax(0)                         # (G,)
        mean_v = flat.mean(0)                           # (G, D)
        best_v = flat.gather(0, best_k.view(1, -1, 1).expand(1, -1, D)).squeeze(0)
        stats['agree'] += int(agree.sum()); stats['disagree'] += int((~agree).sum())
        return torch.where(agree.unsqueeze(-1), mean_v, best_v).reshape(*gshape, D)
    out = _merge_each(client_models, merge)
    tot = stats['agree'] + stats['disagree']
    if tot:
        print(f"    TIES: {stats['agree']}/{tot} gates agree "
              f"({100*stats['agree']/tot:.0f}%), {stats['disagree']} disagree → most confident")
    return out


def fedavg_gate_softconf(client_models, temperature=1.0):
    """Per-gate softmax-confidence weighting; agree gates use uniform mean."""
    stats = {'agree': 0, 'disagree': 0}
    def merge(s):
        K, *gshape, D = s.shape
        flat = s.reshape(K, -1, D)
        votes = flat.argmax(-1); maxl = flat.max(-1).values
        agree = (votes == votes[0:1]).all(0)
        mean_v = flat.mean(0)
        soft_w = F.softmax(maxl / temperature, dim=0).unsqueeze(-1)
        soft_v = (flat * soft_w).sum(0)
        stats['agree'] += int(agree.sum()); stats['disagree'] += int((~agree).sum())
        return torch.where(agree.unsqueeze(-1), mean_v, soft_v).reshape(*gshape, D)
    out = _merge_each(client_models, merge)
    tot = stats['agree'] + stats['disagree']
    if tot:
        print(f"    SoftConf(tau={temperature}): {stats['agree']}/{tot} agree "
              f"({100*stats['agree']/tot:.0f}%), {stats['disagree']} disagree → "
              f"confidence-weighted (softmax)")
    return out


def fedavg_gate_softconf_adaptive(client_models, base_tau=1.0, gap_scale=2.0):
    """Per-gate adaptive softmax-confidence merging.

    For each gate, sets the softmax temperature based on the confidence gap
    between the most confident and second-most confident client:
    - Large gap → low temperature → trust the specialist
    - Small gap → high temperature → average (noise reduction)

    Formula: tau_g = base_tau / (1 + gap_scale * gap_g)

    Data-free, threshold-free, per-gate adaptation.
    """
    keys = [n for n, _ in client_models[0].named_parameters()]
    K = len(client_models)
    avg = {}
    stats = {'agree': 0, 'disagree': 0, 'tau_mean': 0.0, 'tau_min': float('inf'), 'tau_max': 0.0}
    tau_accum = []

    for n in keys:
        stacked = torch.stack([dict(m.named_parameters())[n].data for m in client_models])
        if stacked.shape[-1] in (4, 16) and stacked.dim() >= 3:
            votes = stacked.argmax(dim=-1)
            max_logits = stacked.max(dim=-1).values  # (K, ...)

            result = torch.zeros_like(stacked[0])
            flat_stacked = stacked.reshape(K, -1, stacked.shape[-1])
            flat_votes = votes.reshape(K, -1)
            flat_max = max_logits.reshape(K, -1)
            num_gates = flat_votes.shape[1]

            for g in range(num_gates):
                flat_result = result.reshape(-1, result.shape[-1])
                if (flat_votes[:, g] == flat_votes[0, g]).all():
                    # Agree: uniform average
                    flat_result[g] = flat_stacked[:, g, :].mean(dim=0)
                    stats['agree'] += 1
                else:
                    # Disagree: per-gate adaptive temperature
                    sorted_max, _ = flat_max[:, g].sort(descending=True)
                    gap = (sorted_max[0] - sorted_max[1]).item()
                    tau_g = base_tau / (1.0 + gap_scale * gap)
                    tau_accum.append(tau_g)

                    w = F.softmax(flat_max[:, g] / tau_g, dim=0)  # (K,)
                    flat_result[g] = (flat_stacked[:, g, :] * w.unsqueeze(-1)).sum(dim=0)
                    stats['disagree'] += 1

            avg[n] = result
        else:
            avg[n] = stacked.mean(0)

    total = stats['agree'] + stats['disagree']
    if tau_accum:
        tau_arr = np.array(tau_accum)
        print(f"    SoftConf-Adaptive: {stats['agree']}/{total} agree "
              f"({100*stats['agree']/total:.0f}%), "
              f"tau range [{tau_arr.min():.3f}, {tau_arr.max():.3f}], "
              f"mean={tau_arr.mean():.3f}, median={np.median(tau_arr):.3f}")
    return avg


def fedavg_gate_adaptive(client_models, agreement_threshold=0.08):
    """Adaptive gate merging: switches strategy based on agreement rate.

    1. Compute global agreement rate (fraction of gates where all K clients
       agree on the argmax gate type).
    2. For each gate:
       - Agree: average logits (consensus, noise reduction).
       - Disagree + high agreement: entropy-weighted average (mild non-IID).
       - Disagree + low agreement: most confident client (extreme non-IID).

    Data-free: only uses model parameters. Adapts to heterogeneity regime.
    """
    keys = [n for n, _ in client_models[0].named_parameters()]
    K = len(client_models)

    # First pass: compute agreement rate across all gate parameters
    total_gates = 0
    agree_gates = 0
    for n in keys:
        stacked = torch.stack([dict(m.named_parameters())[n].data for m in client_models])
        if stacked.shape[-1] in (4, 16) and stacked.dim() >= 3:
            votes = stacked.argmax(dim=-1)  # (K, ...)
            flat_votes = votes.reshape(K, -1)
            for g in range(flat_votes.shape[1]):
                total_gates += 1
                if (flat_votes[:, g] == flat_votes[0, g]).all():
                    agree_gates += 1

    agreement_rate = agree_gates / max(total_gates, 1)
    use_avg_for_disagree = agreement_rate > agreement_threshold

    if use_avg_for_disagree:
        disagree_strategy = "entropy-weighted average"
    else:
        disagree_strategy = "most confident client"
    print(f"    Adaptive: {agree_gates}/{total_gates} agree "
          f"({100*agreement_rate:.0f}%), threshold={100*agreement_threshold:.0f}%, "
          f"disagree strategy: {disagree_strategy}")

    # Second pass: merge
    avg = {}
    for n in keys:
        stacked = torch.stack([dict(m.named_parameters())[n].data for m in client_models])
        if stacked.shape[-1] in (4, 16) and stacked.dim() >= 3:
            votes = stacked.argmax(dim=-1)
            confidences = stacked.max(dim=-1).values

            result = torch.zeros_like(stacked[0])
            flat_stacked = stacked.reshape(K, -1, stacked.shape[-1])
            flat_votes = votes.reshape(K, -1)
            flat_conf = confidences.reshape(K, -1)
            num_gates = flat_votes.shape[1]

            for g in range(num_gates):
                gate_votes = flat_votes[:, g]
                flat_result = result.reshape(-1, result.shape[-1])

                if (gate_votes == gate_votes[0]).all():
                    # Agree: uniform average
                    flat_result[g] = flat_stacked[:, g, :].mean(dim=0)
                elif use_avg_for_disagree:
                    # Mild non-IID: entropy-weighted average
                    probs = F.softmax(flat_stacked[:, g, :], dim=-1)
                    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1)  # (K,)
                    weights = F.softmax(-entropy, dim=0)  # (K,)
                    flat_result[g] = (flat_stacked[:, g, :] * weights.unsqueeze(-1)).sum(dim=0)
                else:
                    # Extreme non-IID: most confident client
                    best_k = flat_conf[:, g].argmax().item()
                    flat_result[g] = flat_stacked[best_k, g, :]

            avg[n] = result
        else:
            avg[n] = stacked.mean(0)
    return avg


def fedavg_gate_funcspace(client_models):
    """Function-space FedAvg: log(mean(softmax(logits))). Geom-mean → arithmetic mean."""
    return _merge_each(client_models,
                       lambda s: torch.log(F.softmax(s, dim=-1).mean(0) + 1e-8))


def fedavg_gate_funcspace_conf(client_models, temperature=1.0):
    """Function-space SoftConf: agree → uniform mean of probs; disagree → conf-weighted."""
    K = len(client_models)
    def merge(s):
        votes = s.argmax(-1); maxl = s.max(-1).values
        agree = (votes == votes[0:1]).all(0)
        conf_w = F.softmax(maxl / temperature, dim=0)
        uniform = torch.full_like(conf_w, 1.0 / K)
        w = torch.where(agree.unsqueeze(0), uniform, conf_w)
        return torch.log((F.softmax(s, dim=-1) * w.unsqueeze(-1)).sum(0) + 1e-8)
    return _merge_each(client_models, merge)


def get_last_layer_outputs(model, loader, device, binarize=True, flat=True):
    """Collect hard gate outputs from the last LogicLayer (before GroupSum).

    Returns (gate_outputs, labels) where gate_outputs is (N, num_gates) binary.
    """
    model.eval()
    logic_types = (LogicLayer,)
    if HAS_TORCHLOGIX:
        logic_types = (LogicLayer, TLLogicDense)
    layers = [m for m in model.modules() if isinstance(m, logic_types)]
    last = layers[-1]
    outs, ys = [], []
    def hook(m, inp, out):
        outs.append(out.detach().cpu())  # always move to CPU to avoid OOM
    h = last.register_forward_hook(hook)
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            if flat:
                x = x.view(x.size(0), -1)
            if binarize:
                x = (x > 0.5).float()
            model(x)
            ys.append(y)
    h.remove()
    return torch.cat(outs), torch.cat(ys)


def boolean_feature_selection(models, probe_loader, test_loader, device,
                              num_classes=10, num_select=50, binarize=True, flat=True):
    """Pool boolean gates from all models, select most discriminative per class.

    For each class, ranks all K*num_gates gates by discriminative power
    (mean activation difference: in-class vs out-of-class) on probe data,
    selects the top num_select, and evaluates on test data using GroupSum
    over the selected gates.

    Returns (accuracy, selected_indices_per_class).
    """
    # Collect gate outputs on probe and test data (last layer, K models concatenated)
    pgs, tgs = zip(*[(get_last_layer_outputs(m, probe_loader, device, binarize=binarize, flat=flat),
                      get_last_layer_outputs(m, test_loader, device, binarize=binarize, flat=flat))
                     for m in models])
    labels = pgs[0][1].cpu(); test_labels = tgs[0][1].cpu()
    probe_gates = torch.cat([p for p, _ in pgs], dim=1).float().cpu()
    test_gates = torch.cat([t for t, _ in tgs], dim=1).float().cpu()

    # Score each gate per class: E[g | y=c] − E[g | y!=c]
    scores = torch.stack([probe_gates[labels == c].mean(0) - probe_gates[labels != c].mean(0)
                          for c in range(num_classes)])
    selected = {c: scores[c].argsort(descending=True)[:num_select] for c in range(num_classes)}
    logits = torch.stack([test_gates[:, selected[c]].sum(1) for c in range(num_classes)], dim=1)
    acc = (logits.argmax(1) == test_labels).float().mean().item()
    return acc, selected


def boolean_feature_selection_diverse(models, probe_loader, test_loader, device,
                                      num_classes=10, num_select=50, binarize=True,
                                      flat=True, diversity_lambda=0.5):
    """Fourier-diverse BFS: greedy selection with diversity penalty.

    Instead of taking top-n by score (which may select redundant gates),
    greedily selects gates that maximize:
        adjusted_score = score(g, c) - lambda * max_correlation_with_selected

    The correlation between gates is computed from their output vectors on
    probe data — functionally equivalent to the Fourier inner product.
    """
    all_probe, all_test = [], []
    labels = test_labels = None
    for m in models:
        pg, labels = get_last_layer_outputs(m, probe_loader, device, binarize=binarize, flat=flat)
        all_probe.append(pg)
        tg, test_labels = get_last_layer_outputs(m, test_loader, device, binarize=binarize, flat=flat)
        all_test.append(tg)

    probe_gates = torch.cat(all_probe, dim=1).float()  # (N_probe, K*num_gates)
    test_gates = torch.cat(all_test, dim=1).float()

    G = probe_gates.shape[1]

    # Move to CPU for selection (avoids device issues)
    probe_cpu = probe_gates.cpu()
    test_cpu = test_gates.cpu()
    labels_cpu = labels.cpu()

    # Subsample probe data for correlation matrix (5K samples suffices for diversity)
    N_probe = probe_cpu.shape[0]
    N_sub = min(5000, N_probe)
    perm = torch.randperm(N_probe)[:N_sub]
    probe_sub = probe_cpu[perm]

    # Center and normalize for correlation computation
    probe_centered = probe_sub - probe_sub.mean(0, keepdim=True)
    norms = probe_centered.norm(dim=0, keepdim=True).clamp(min=1e-8)
    probe_normed = probe_centered / norms  # (N_sub, G)

    # Pre-compute full correlation matrix (G, G)
    corr_matrix = (probe_normed.T @ probe_normed).abs()  # (G, G)

    # Score each gate for each class
    scores = torch.zeros(num_classes, G)
    for c in range(num_classes):
        in_mask = (labels_cpu == c)
        scores[c] = probe_cpu[in_mask].mean(0) - probe_cpu[~in_mask].mean(0)

    # Greedy selection with diversity penalty
    selected = {}
    logits = torch.zeros(test_cpu.shape[0], num_classes)
    for c in range(num_classes):
        sel_indices = []
        max_corr_with_selected = torch.zeros(G)  # track max corr incrementally

        for _ in range(num_select):
            # Find best gate: score - lambda * max_correlation
            adj_scores = scores[c] - diversity_lambda * max_corr_with_selected
            # Mask already selected
            for s in sel_indices:
                adj_scores[s] = -float('inf')

            best_idx = adj_scores.argmax().item()
            sel_indices.append(best_idx)
            # Update max correlations incrementally
            max_corr_with_selected = torch.max(
                max_corr_with_selected, corr_matrix[:, best_idx])

        selected[c] = torch.tensor(sel_indices)
        logits[:, c] = test_cpu[:, sel_indices].sum(dim=1)

    acc = (logits.argmax(1) == test_labels.cpu()).float().mean().item()
    return acc, selected


def bfs_federated(models, client_loaders, test_loader, device,
                  num_classes=10, num_select=50, binarize=True, flat=True,
                  return_comm=False, dp_sigma=0.0):
    """Federated BFS without a central probe set.

    With dp_sigma > 0, adds per-coordinate Gaussian noise N(0, dp_sigma^2)
    to every entry of the upload (in_sum, in_count, total_sum, total_count)
    before the server aggregates.  This is a per-coordinate noise stress
    test, NOT a calibrated (epsilon, delta)-DP mechanism: the release is
    the joint upload vector, whose L2 sensitivity per record is bounded
    by sqrt(2 K N + 2) (a removed example can flip up to KN gate outputs
    in in_sum and again in total_sum, plus two count entries).  A
    formally-private deployment would clip per-record contributions and
    calibrate dp_sigma against that vector sensitivity.

    Mathematically equivalent to boolean_feature_selection() with the union of
    client data as the probe, but computed via aggregated per-(class, gate)
    statistics. No labeled data leaves any client.

    Pipeline:
      1. Each client downloads all K hardened models (one-shot exchange).
      2. Each client k evaluates every model's last layer on its own data,
         producing local gate outputs g^(k) of shape (n_k, K * N).
      3. Each client computes per-class sums of g^(k) and per-class counts,
         uploads them to the server.
      4. Server reconstructs the global discriminative score
            s_{c,j} = sum_k in_sum_k[c,j] / sum_k in_count_k[c]
                     - sum_k out_sum_k[c,j] / sum_k out_count_k[c]
         and selects the top num_select gates per class.

    Returns (accuracy, selected) matching boolean_feature_selection's signature.
    With return_comm=True, also returns a dict of communication costs.
    """
    K = len(models)
    in_sum = None       # (C, K*N) — accumulated across clients
    in_count = torch.zeros(num_classes)  # float for DP-noise compatibility
    total_sum = None    # (K*N,) — accumulated across clients
    total_count = 0.0

    for k_idx, loader in enumerate(client_loaders):
        per_model_gates = []
        labels_k = None
        for m in models:
            gates_m, labels_k = get_last_layer_outputs(
                m, loader, device, binarize=binarize, flat=flat)
            per_model_gates.append(gates_m)
        gates_k = torch.cat(per_model_gates, dim=1).float().cpu()  # (n_k, K*N)
        labels_k = labels_k.cpu()

        if in_sum is None:
            KN = gates_k.shape[1]
            in_sum = torch.zeros(num_classes, KN)
            total_sum = torch.zeros(KN)

        # Per-client uploads, computed locally before "leaving" the device.
        upload_in_sum = torch.zeros(num_classes, gates_k.shape[1])
        upload_in_count = torch.zeros(num_classes)
        for c in range(num_classes):
            mask = (labels_k == c)
            if mask.any():
                upload_in_sum[c] = gates_k[mask].sum(dim=0)
                upload_in_count[c] = float(mask.sum().item())
        upload_total_sum = gates_k.sum(dim=0)
        upload_total_count = float(gates_k.shape[0])

        # Gaussian-mechanism DP applied per-client to the upload payload.
        if dp_sigma > 0:
            upload_in_sum = upload_in_sum + dp_sigma * torch.randn_like(upload_in_sum)
            upload_in_count = upload_in_count + dp_sigma * torch.randn_like(upload_in_count)
            upload_total_sum = upload_total_sum + dp_sigma * torch.randn_like(upload_total_sum)
            upload_total_count = upload_total_count + dp_sigma * float(torch.randn(1).item())

        in_sum += upload_in_sum
        in_count += upload_in_count
        total_sum += upload_total_sum
        total_count += upload_total_count

    # Recover out-of-class statistics from totals (no extra communication needed)
    out_sum = total_sum.unsqueeze(0).expand(num_classes, -1) - in_sum  # (C, K*N)
    out_count = total_count - in_count

    # Score = E[g | y=c] - E[g | y!=c]
    scores = torch.zeros(num_classes, in_sum.shape[1])
    for c in range(num_classes):
        ic, oc = float(in_count[c]), float(out_count[c])
        if ic > 0 and oc > 0:
            scores[c] = in_sum[c] / ic - out_sum[c] / oc
        elif ic > 0:
            scores[c] = in_sum[c] / ic
        # else: no class-c samples (or DP-noise wiped them) — leave at zero

    # Server: run all K models on test data, classify with selected gates
    all_test = []
    test_labels = None
    for m in models:
        tg, test_labels = get_last_layer_outputs(
            m, test_loader, device, binarize=binarize, flat=flat)
        all_test.append(tg)
    test_gates = torch.cat(all_test, dim=1).float().cpu()
    test_labels = test_labels.cpu()

    selected = {}
    logits = torch.zeros(test_gates.shape[0], num_classes)
    for c in range(num_classes):
        top = scores[c].argsort(descending=True)[:num_select]
        selected[c] = top
        logits[:, c] = test_gates[:, top].sum(dim=1)
    acc = (logits.argmax(1) == test_labels).float().mean().item()

    if return_comm:
        # Per-client comm cost (worst case, star topology):
        #   model exchange: 1 upload + (K-1) downloads ≈ K * |gate_logits|
        #   stats upload:   in_sum (C * KN floats) + in_count (C ints)
        # Out-of-class stats are derived at server from total_sum, which itself
        # equals sum_c in_sum[c, :] — so no extra channel needed.
        model_logit_bytes = sum(
            p.numel() * p.element_size() for p in models[0].parameters())
        stats_bytes = (num_classes * in_sum.shape[1] * 4  # in_sum, float32
                       + num_classes * 8)                 # in_count, int64
        per_client_comm = K * model_logit_bytes + stats_bytes
        comm = {
            'model_exchange_bytes_per_client': K * model_logit_bytes,
            'statistics_bytes_per_client': stats_bytes,
            'total_bytes_per_client': per_client_comm,
            'K': K,
        }
        return acc, selected, comm
    return acc, selected


def bfs_federated_dp(models, client_loaders, test_loader, device,
                     num_classes=10, num_select=50, binarize=True, flat=True,
                     clip_norm=1.0, epsilon=1.0, delta=1e-5):
    """Federated BFS with calibrated (epsilon, delta)-DP per upload.

    Per-record contributions are clipped to L2 norm `clip_norm`, summed
    locally, and the per-client upload is released with Gaussian noise
    initialized at clip_norm * sqrt(2 ln(1.25/delta)) / epsilon. If its
    exact Gaussian divergence exceeds delta, sigma is increased by binary
    search. Independent noise is added to every coordinate of the clipped sum.

    Under add/remove record adjacency, each example contributes to exactly one
    client's upload, so the
    cross-client aggregate enjoys the same per-record privacy by parallel
    composition.  No extra accountant needed (single-shot release).

    The contribution vector v_i for record (x_i, y_i = c_i) has entries
        in_sum[c_i, :] = g(x_i),     0 elsewhere in in_sum
        in_count[c_i] = 1,           0 elsewhere in in_count
        total_sum    = g(x_i)
        total_count  = 1
    so ||v_i||_2 = sqrt(2 * ||g(x_i)||_2^2 + 2).  We compute this per
    record and apply min(1, C/||v_i||) as the clipping factor.
    """
    K = len(models)
    if math.isinf(epsilon):
        sigma = 0.0
    else:
        sigma = (
            clip_norm
            * math.sqrt(2 * math.log(1.25 / delta))
            / max(epsilon, 1e-12)
        )

        def achieved_delta(noise):
            mu = clip_norm / noise
            cdf = lambda value: 0.5 * math.erfc(-value / math.sqrt(2))
            return (
                cdf(mu / 2 - epsilon / mu)
                - math.exp(epsilon) * cdf(-mu / 2 - epsilon / mu)
            )

        if achieved_delta(sigma) > delta:
            low, high = sigma, 2 * sigma
            while achieved_delta(high) > delta:
                low, high = high, 2 * high
            for _ in range(80):
                midpoint = 0.5 * (low + high)
                if achieved_delta(midpoint) > delta:
                    low = midpoint
                else:
                    high = midpoint
            sigma = high

    in_sum = None  # (C, KN), on GPU, lazily-built
    in_count = torch.zeros(num_classes, device=device)
    total_sum = None
    total_count = 0.0
    KN = None
    seg_starts = None  # cumulative gate-index offsets, one per model

    for loader in client_loaders:
        # ---- Pass 1: collect per-model outputs on CPU; compute global norms.
        per_model_cpu = []
        labels_k = None
        for m in models:
            gates_m, labels_k = get_last_layer_outputs(
                m, loader, device, binarize=binarize, flat=flat)
            per_model_cpu.append(gates_m)               # CPU tensor, (n_k, N_m)
        if seg_starts is None:
            sizes = [g.shape[1] for g in per_model_cpu]
            seg_starts = [sum(sizes[:i]) for i in range(len(sizes) + 1)]
            KN = seg_starts[-1]
            in_sum = torch.zeros(num_classes, KN, device=device)
            total_sum = torch.zeros(KN, device=device)

        n_k = labels_k.shape[0]
        labels_dev = labels_k.to(device)
        firing_norm_sq = torch.zeros(n_k, device=device)
        for gates_m in per_model_cpu:
            gm = gates_m.float().to(device)
            firing_norm_sq += (gm * gm).sum(dim=1)
            del gm
        record_norm = (2.0 * firing_norm_sq + 2.0).clamp(min=1e-12).sqrt()
        clip = (clip_norm / record_norm).clamp(max=1.0)              # (n_k,)
        del firing_norm_sq, record_norm

        one_hot = torch.zeros(n_k, num_classes, device=device)
        one_hot.scatter_(1, labels_dev.unsqueeze(1).long(), 1.0)

        # ---- Pass 2: apply clip per-model; accumulate into the right segment.
        oh_clipped = one_hot * clip.unsqueeze(1)                     # (n_k, C)
        for k_idx, gates_m in enumerate(per_model_cpu):
            gm = gates_m.float().to(device)
            gm_clipped = gm * clip.unsqueeze(1)
            seg = slice(seg_starts[k_idx], seg_starts[k_idx + 1])
            upload_in_sum_seg = one_hot.t() @ gm_clipped              # (C, N_m)
            upload_total_sum_seg = gm_clipped.sum(dim=0)              # (N_m,)
            if sigma > 0:
                upload_in_sum_seg = upload_in_sum_seg + sigma * torch.randn_like(upload_in_sum_seg)
                upload_total_sum_seg = upload_total_sum_seg + sigma * torch.randn_like(upload_total_sum_seg)
            in_sum[:, seg] += upload_in_sum_seg
            total_sum[seg] += upload_total_sum_seg
            del gm, gm_clipped
            torch.cuda.empty_cache() if device.type == 'cuda' else None

        upload_in_count = oh_clipped.sum(dim=0)                       # (C,)
        upload_total_count = float(clip.sum().item())
        if sigma > 0:
            upload_in_count = upload_in_count + sigma * torch.randn_like(upload_in_count)
            upload_total_count = upload_total_count + sigma * float(torch.randn(1).item())
        in_count += upload_in_count
        total_count += upload_total_count

        del per_model_cpu, one_hot, oh_clipped, clip, labels_dev
        torch.cuda.empty_cache() if device.type == 'cuda' else None

    out_sum = total_sum.unsqueeze(0).expand(num_classes, -1) - in_sum
    out_count = total_count - in_count
    scores = torch.zeros(num_classes, in_sum.shape[1], device=device)
    for c in range(num_classes):
        ic, oc = float(in_count[c]), float(out_count[c])
        if abs(ic) > 1e-6 and abs(oc) > 1e-6:
            scores[c] = in_sum[c] / ic - out_sum[c] / oc
        elif abs(ic) > 1e-6:
            scores[c] = in_sum[c] / ic

    # Server runs all K models on test, classifies via selected gates.
    selected = {}
    for c in range(num_classes):
        selected[c] = scores[c].argsort(descending=True)[:num_select].cpu()

    # Streaming test eval: process each client model's segment, sum the
    # selected gates per class without ever materialising the full test
    # gate matrix.
    n_test = None
    logits = None
    test_labels = None
    for k_idx, m in enumerate(models):
        tg, test_labels = get_last_layer_outputs(
            m, test_loader, device, binarize=binarize, flat=flat)
        tg = tg.float().to(device)
        if logits is None:
            n_test = tg.shape[0]
            logits = torch.zeros(n_test, num_classes, device=device)
        seg = slice(seg_starts[k_idx], seg_starts[k_idx + 1])
        seg_size = seg_starts[k_idx + 1] - seg_starts[k_idx]
        for c in range(num_classes):
            sel_c = selected[c]
            mask = (sel_c >= seg_starts[k_idx]) & (sel_c < seg_starts[k_idx + 1])
            local_idx = (sel_c[mask] - seg_starts[k_idx]).to(device)
            if local_idx.numel() > 0:
                logits[:, c] += tg[:, local_idx].sum(dim=1)
        del tg
        torch.cuda.empty_cache() if device.type == 'cuda' else None
    test_labels = test_labels.to(device)
    acc = (logits.argmax(1) == test_labels).float().mean().item()
    return acc, selected


def bfs_federated_diverse(models, client_loaders, test_loader, device,
                          num_classes=10, num_select=50, binarize=True, flat=True,
                          diversity_lambda=0.3, shortlist_per_class=200,
                          return_comm=False):
    """Two-round federated diverse BFS, math-equivalent to centralised
    diverse BFS on the union of client data.

    Round 1 -- first moments. Each client uploads per-class gate sums and
    counts (same as bfs_federated).  Server scores all gates and builds a
    shortlist (top-M per class, unioned).

    Round 2 -- second moments on shortlist.  Each client uploads, for the
    shortlist gates only, the matrix sum_x g_i(x) g_j(x) plus single-gate
    sums.  Server reconstructs the union-data correlation matrix on the
    shortlist exactly and runs the diversity-aware greedy.

    Bandwidth scales as O(K * |shortlist|^2), independent of total pool G.
    """
    K = len(models)

    # ---- Round 1: collect per-client gates and first moments. ----
    # We hold each client's gates in CPU mem to avoid re-evaluating models
    # for round 2; for MNIST FC at K=5 this is ~50K x 1500 floats ~= 0.3 GB.
    per_client_gates = []
    in_sum = None
    in_count = torch.zeros(num_classes)
    total_sum = None
    total_count = 0.0

    for loader in client_loaders:
        per_model_gates = []
        labels_k = None
        for m in models:
            gates_m, labels_k = get_last_layer_outputs(
                m, loader, device, binarize=binarize, flat=flat)
            per_model_gates.append(gates_m)
        gates_k = torch.cat(per_model_gates, dim=1).float().cpu()
        labels_k = labels_k.cpu()
        per_client_gates.append((gates_k, labels_k))

        if in_sum is None:
            KN = gates_k.shape[1]
            in_sum = torch.zeros(num_classes, KN)
            total_sum = torch.zeros(KN)

        for c in range(num_classes):
            mask = (labels_k == c)
            if mask.any():
                in_sum[c] += gates_k[mask].sum(dim=0)
                in_count[c] += float(mask.sum().item())
        total_sum += gates_k.sum(dim=0)
        total_count += float(gates_k.shape[0])

    out_sum = total_sum.unsqueeze(0).expand(num_classes, -1) - in_sum
    out_count = total_count - in_count

    G = in_sum.shape[1]
    scores = torch.zeros(num_classes, G)
    for c in range(num_classes):
        ic, oc = float(in_count[c]), float(out_count[c])
        if ic > 0 and oc > 0:
            scores[c] = in_sum[c] / ic - out_sum[c] / oc
        elif ic > 0:
            scores[c] = in_sum[c] / ic

    # ---- Build shortlist: top-M per class, union. ----
    shortlist_set = set()
    M = min(shortlist_per_class, G)
    for c in range(num_classes):
        top = scores[c].argsort(descending=True)[:M].tolist()
        shortlist_set.update(top)
    shortlist = sorted(shortlist_set)
    sl_idx = torch.tensor(shortlist, dtype=torch.long)
    Ms = len(shortlist)

    # ---- Round 2: pair sums on shortlist. ----
    pair_sum = torch.zeros(Ms, Ms)
    single_sum = torch.zeros(Ms)
    n_total = 0.0
    for gates_k, _ in per_client_gates:
        gks = gates_k[:, sl_idx]
        pair_sum += gks.t() @ gks
        single_sum += gks.sum(dim=0)
        n_total += float(gks.shape[0])

    means = single_sum / n_total
    cov = pair_sum / n_total - means.unsqueeze(0) * means.unsqueeze(1)
    # Binary gates: var(g) = E[g] - E[g]^2.  For non-binary (soft) outputs
    # this still holds in expectation only approximately.  We clamp.
    var = (means - means * means).clamp(min=1e-8)
    std = var.sqrt()
    corr = (cov / (std.unsqueeze(0) * std.unsqueeze(1))).abs()

    # ---- Greedy diverse selection on shortlist. ----
    scores_sl = scores[:, sl_idx]
    selected = {}
    for c in range(num_classes):
        sel_local = []
        max_corr = torch.zeros(Ms)
        for _ in range(num_select):
            adj = scores_sl[c] - diversity_lambda * max_corr
            for s in sel_local:
                adj[s] = -float('inf')
            best = int(adj.argmax().item())
            sel_local.append(best)
            max_corr = torch.max(max_corr, corr[:, best])
        selected[c] = sl_idx[torch.tensor(sel_local, dtype=torch.long)]

    # ---- Eval on test. ----
    all_test = []
    test_labels = None
    for m in models:
        tg, test_labels = get_last_layer_outputs(
            m, test_loader, device, binarize=binarize, flat=flat)
        all_test.append(tg)
    test_gates = torch.cat(all_test, dim=1).float().cpu()
    test_labels = test_labels.cpu()

    logits = torch.zeros(test_gates.shape[0], num_classes)
    for c in range(num_classes):
        logits[:, c] = test_gates[:, selected[c]].sum(dim=1)
    acc = (logits.argmax(1) == test_labels).float().mean().item()

    if return_comm:
        model_logit_bytes = sum(
            p.numel() * p.element_size() for p in models[0].parameters())
        round1_bytes = num_classes * G * 4 + num_classes * 4
        round2_bytes = Ms * Ms * 4 + Ms * 4
        comm = {
            'model_exchange_bytes_per_client': K * model_logit_bytes,
            'round1_stats_bytes': round1_bytes,
            'round2_stats_bytes': round2_bytes,
            'total_bytes_per_client': K * model_logit_bytes + round1_bytes + round2_bytes,
            'shortlist_size': Ms,
            'K': K,
        }
        return acc, selected, comm
    return acc, selected


def get_client_classes(dataset, indices):
    """Return sorted list of unique classes and a label remap tensor."""
    targets = dataset.targets
    classes = sorted(set(int(targets[i]) for i in indices))
    remap = torch.full((10,), -1, dtype=torch.long)
    for local_idx, global_cls in enumerate(classes):
        remap[global_cls] = local_idx
    return classes, remap


# -- Misc helpers --------------------------------------------------------------

def data_iter(loader):
    """Infinite iterator over a DataLoader."""
    while True:
        yield from loader


GATES = ['FALSE', 'AND', 'A>B', 'A', 'B>A', 'B', 'XOR', 'OR',
         'NOR', 'XNOR', 'NOT_B', 'B=>A', 'NOT_A', 'A=>B', 'NAND', 'TRUE']


def print_gate_stats(model, label=""):
    counts = torch.zeros(16, dtype=torch.long)
    for p in model.parameters():
        if p.shape[-1] in (4, 16):
            for g in p.data.argmax(-1).flatten():
                counts[g] += 1
    header = f"Gate Statistics ({label})" if label else "Gate Statistics"
    print(f"\n=== {header} ===")
    for g in range(16):
        if counts[g] > 0:
            print(f"  {GATES[g]:8s}: {counts[g]}")
