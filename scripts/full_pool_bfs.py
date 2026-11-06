"""Unsubsampled BFS on LogicTreeNet-B (Paper-B) checkpoints.

Paper-B has K=5 clients with ~320K gates each (~1.6M total).  The
standard BFS pipeline materialises a (N_probe x 1.6M) gate matrix for
scoring and a (N_test x 1.6M) matrix for evaluation.  At float32 with
N_test=10K that's 64 GB just for test_gates -- impossible on the 32 GB
RTX 5090 we used to subsample to 5K gates per client (1.5% of pool).
This script runs the un-subsampled version on a 96 GB RTX Pro 6000
to test whether the ~5-pt gap to the ensemble at Paper-B scale is a
budget artefact or a real ceiling.

For diverse BFS we cannot materialise the full GxG correlation matrix
(1.6M^2 x 4 bytes ~= 10 TB).  Instead we keep probe_normed on the GPU
and recompute the (1, G) correlation row for each newly selected gate
at the moment we need it -- O(N_probe x G) per selection, microseconds
on Pro 6000 hardware.

Output: scripts/results/paper_b_unsubsampled_bfs.json
Done sentinel: scripts/results/paper_b_unsubsampled_bfs.DONE
"""
import argparse, json, os, sys, time
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'federated'))
from common import get_cifar10, get_last_layer_outputs


def standard_bfs(probe_gates, test_gates, labels, test_labels,
                 num_classes=10, num_select=100):
    G = probe_gates.shape[1]
    scores = torch.zeros(num_classes, G, device=probe_gates.device)
    for c in range(num_classes):
        in_mask = (labels == c)
        scores[c] = probe_gates[in_mask].float().mean(0) - probe_gates[~in_mask].float().mean(0)

    selected = {}
    logits = torch.zeros(test_gates.shape[0], num_classes, device=test_gates.device)
    for c in range(num_classes):
        top = scores[c].argsort(descending=True)[:num_select]
        selected[c] = top
        logits[:, c] = test_gates[:, top].float().sum(1)
    acc = (logits.argmax(1) == test_labels).float().mean().item()
    return acc, scores, selected


def diverse_bfs_streaming(probe_gates, test_gates, scores, test_labels,
                          num_classes=10, num_select=100, diversity_lambda=0.3):
    """Diverse BFS without materialising the GxG correlation matrix.

    For each class we run greedy selection.  When a gate `j` is newly
    selected, we compute its correlation row corr_j = |probe_normed^T
    @ probe_normed[:, j]| (shape (G,)) and update the running max-corr
    vector.  The next pick maximises score - lambda * max_corr.
    """
    device = probe_gates.device
    G = probe_gates.shape[1]

    # Center + normalise probe along the sample axis (in-place to save mem).
    probe_centered = probe_gates.float() - probe_gates.float().mean(0, keepdim=True)
    norms = probe_centered.norm(dim=0, keepdim=True).clamp(min=1e-8)
    probe_normed = probe_centered / norms  # (N_probe, G)
    del probe_centered

    div_logits = torch.zeros(test_gates.shape[0], num_classes, device=device)
    for c in range(num_classes):
        sel = []
        max_corr = torch.zeros(G, device=device)
        for k in range(num_select):
            adj = scores[c] - diversity_lambda * max_corr
            for s in sel:
                adj[s] = -float('inf')
            best = adj.argmax().item()
            sel.append(best)
            # Correlation of `best` against all gates: (N_probe,) . (N_probe, G) -> (G,)
            corr_row = (probe_normed[:, best:best+1].T @ probe_normed).abs().squeeze(0)
            max_corr = torch.max(max_corr, corr_row)
        sel_idx = torch.tensor(sel, device=device, dtype=torch.long)
        div_logits[:, c] = test_gates[:, sel_idx].float().sum(1)
    acc = (div_logits.argmax(1) == test_labels).float().mean().item()
    return acc


def collect_gates(checkpoint_dir, probe_loader, test_loader, device, num_clients=5):
    """Stream client checkpoints; concat their last-layer outputs along gate axis."""
    all_probe, all_test = [], []
    labels = test_labels = None
    for i in range(num_clients):
        m = torch.load(f'{checkpoint_dir}/client_{i}.pt',
                       map_location='cpu', weights_only=False)
        m = m.to(device)
        pg, labels = get_last_layer_outputs(m, probe_loader, device,
                                            binarize=False, flat=False)
        tg, test_labels = get_last_layer_outputs(m, test_loader, device,
                                                 binarize=False, flat=False)
        all_probe.append(pg)
        all_test.append(tg)
        m.cpu(); del m
        torch.cuda.empty_cache()
        print(f'    client {i}: gates={pg.shape[1]:,} '
              f'(running total {sum(p.shape[1] for p in all_probe):,})',
              flush=True)
    probe_gates = torch.cat(all_probe, dim=1)
    test_gates = torch.cat(all_test, dim=1)
    del all_probe, all_test
    return probe_gates, test_gates, labels, test_labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alphas', nargs='+', default=['0.5', '0.1'])
    ap.add_argument('--num_clients', type=int, default=5)
    ap.add_argument('--num_select', type=int, default=100)
    ap.add_argument('--diversity_lambda', type=float, default=0.3)
    ap.add_argument('--probe_size', type=int, default=1000)
    ap.add_argument('--checkpoint_root', default='checkpoints/runs')
    ap.add_argument('--out', default='scripts/results/paper_b_unsubsampled_bfs.json')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f'GPU: {torch.cuda.get_device_name()}, '
              f'{props.total_memory / 1e9:.1f} GB total')

    train_ds, test_ds = get_cifar10(flatten=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=200, shuffle=False)
    g = torch.Generator().manual_seed(0)
    probe_idx = torch.randperm(len(train_ds), generator=g)[:args.probe_size].tolist()
    probe_ds = torch.utils.data.Subset(train_ds, probe_idx)
    probe_loader = torch.utils.data.DataLoader(probe_ds, batch_size=200, shuffle=False)

    results = {}
    t_start = time.time()

    for alpha in args.alphas:
        ckpt_dir = (f'{args.checkpoint_root}/paper_b_a{alpha}_full/'
                    f'b1_cifar10_dirichlet_K{args.num_clients}')
        if not os.path.isdir(ckpt_dir):
            ckpt_dir = (f'{args.checkpoint_root}/paper_b_a{alpha}/'
                        f'b1_cifar10_dirichlet_K{args.num_clients}')
        if not os.path.isdir(ckpt_dir):
            print(f'  alpha={alpha}: no checkpoint dir, skipping (looked at {ckpt_dir})')
            continue

        print(f'\n{"="*60}\n  Paper-B alpha={alpha} unsubsampled BFS')
        print(f'  ckpt: {ckpt_dir}\n{"="*60}', flush=True)
        t0 = time.time()

        probe_gates, test_gates, labels, test_labels = collect_gates(
            ckpt_dir, probe_loader, test_loader, device, num_clients=args.num_clients)

        G = probe_gates.shape[1]
        N_probe, N_test = probe_gates.shape[0], test_gates.shape[0]
        print(f'  pool: G={G:,} gates, probe={N_probe} test={N_test}')
        print(f'  probe_gates: {probe_gates.element_size() * probe_gates.numel() / 1e9:.1f} GB '
              f'on {probe_gates.device}  -> moving to {device}')
        print(f'  test_gates : {test_gates.element_size() * test_gates.numel() / 1e9:.1f} GB '
              f'on {test_gates.device}  -> moving to {device}', flush=True)

        # All BFS ops happen on GPU at this scale (CPU would be ~minutes per op).
        probe_gates = probe_gates.to(device)
        test_gates = test_gates.to(device)
        labels = labels.to(device)
        test_labels = test_labels.to(device)
        print(f'  GPU mem after move: '
              f'{torch.cuda.memory_allocated()/1e9:.1f} GB allocated, '
              f'{torch.cuda.memory_reserved()/1e9:.1f} GB reserved', flush=True)

        # Standard BFS
        bfs_acc, scores, _ = standard_bfs(
            probe_gates, test_gates, labels, test_labels,
            num_select=args.num_select)
        print(f'  BFS         (n={args.num_select}): {100*bfs_acc:.2f}%', flush=True)

        # Diverse BFS streaming
        div_acc = diverse_bfs_streaming(
            probe_gates, test_gates, scores, test_labels,
            num_select=args.num_select, diversity_lambda=args.diversity_lambda)
        print(f'  Diverse BFS (n={args.num_select}, lambda={args.diversity_lambda}): '
              f'{100*div_acc:.2f}%', flush=True)

        results[alpha] = {
            'gates_pool': int(G),
            'n_probe': int(N_probe),
            'n_test': int(N_test),
            'num_select': args.num_select,
            'bfs_acc': float(bfs_acc),
            'diverse_bfs_acc': float(div_acc),
            'diversity_lambda': args.diversity_lambda,
            'elapsed_sec': time.time() - t0,
        }

        # Persist after each alpha so partial results survive a crash.
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump({'config': vars(args), 'results': results}, f, indent=2)

        del probe_gates, test_gates, scores
        torch.cuda.empty_cache()
        print(f'  alpha={alpha} done in {(time.time()-t0)/60:.1f} min '
              f'(total {(time.time()-t_start)/60:.1f} min)', flush=True)

    sentinel = args.out.rsplit('.', 1)[0] + '.DONE'
    with open(sentinel, 'w') as f:
        f.write(f'completed in {(time.time()-t_start)/60:.1f} min\n')

    print(f'\nWrote {args.out}')
    print(f'\nSummary:')
    for alpha, r in results.items():
        print(f'  alpha={alpha}: BFS={100*r["bfs_acc"]:.2f}% '
              f'Diverse={100*r["diverse_bfs_acc"]:.2f}%')


if __name__ == '__main__':
    main()
