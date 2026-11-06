"""Multi-seed BFS sweep across alpha values, K=5, MNIST FC.

For each (alpha, seed) we train K=5 clients on a Dirichlet partition with
an 80/20 split inside each client (80% used for training, 20% held out
for the federated selection step).  We then evaluate five aggregation
strategies on the same trained clients:

  bfs_probe         -- probe-based standard BFS (server has labelled probe).
  bfs_fed           -- federated standard BFS using each client's TRAINING
                       data as the selection set (current paper).
  bfs_fed_holdout   -- federated standard BFS using each client's HELD-OUT
                       20% as the selection set.  Tests for selection-side
                       overfitting.
  bfs_fed_diverse   -- two-round federated diverse BFS (math-equivalent
                       to centralised diverse BFS on the union of client
                       data).  Uses held-out selection data.
  diverse_probe     -- probe-based diverse BFS.

Reports mean +/- std across seeds per (alpha, method).

Output: scripts/results/multiseed_alpha_bfs.json
Done sentinel: scripts/results/multiseed_alpha_bfs.DONE
"""
import argparse, copy, json, os, random, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'federated'))
from common import (
    make_difflogic_net, get_mnist, partition_dirichlet, evaluate,
    train_one_client,
    boolean_feature_selection, boolean_feature_selection_diverse,
    bfs_federated, bfs_federated_diverse,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alphas', nargs='+', type=float,
                    default=[0.01, 0.1, 0.5, 1.0, 5.0])
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    ap.add_argument('--num_clients', type=int, default=5)
    ap.add_argument('--num_iters', type=int, default=500)
    ap.add_argument('--num_select', type=int, default=100)
    ap.add_argument('--diversity_lambda', type=float, default=0.3)
    ap.add_argument('--holdout_frac', type=float, default=0.2)
    ap.add_argument('--out', default='scripts/results/multiseed_alpha_bfs.json')
    args = ap.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f'Device: {device}; CUDA: {torch.cuda.is_available()}', flush=True)

    train_ds, test_ds = get_mnist()
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False)
    probe_loader = torch.utils.data.DataLoader(train_ds, batch_size=1000, shuffle=False)

    method_keys = ['bfs_probe', 'bfs_fed', 'bfs_fed_holdout',
                   'bfs_fed_diverse', 'diverse_probe']
    results = {str(a): {m: [] for m in method_keys + ['client_mean']}
               for a in args.alphas}
    t_start = time.time()

    for alpha in args.alphas:
        for seed in args.seeds:
            t0 = time.time()
            torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
            global_model = make_difflogic_net(784, 500, 3, 10, tau=10.0)

            client_idx = partition_dirichlet(train_ds, args.num_clients, alpha=alpha)
            for k in range(args.num_clients):
                if len(client_idx[k]) == 0:
                    donor = max(range(args.num_clients), key=lambda i: len(client_idx[i]))
                    client_idx[k] = [client_idx[donor][0]]

            # 80/20 split per client (deterministic per seed via torch.randperm)
            train_idx_per_client = []
            holdout_idx_per_client = []
            for idx_list in client_idx:
                idx_t = torch.tensor(idx_list)
                n = len(idx_t)
                shuffle = torch.randperm(n, generator=torch.Generator().manual_seed(seed * 1000 + n))
                idx_t = idx_t[shuffle]
                n_h = max(1, int(round(args.holdout_frac * n)))
                holdout_idx_per_client.append(idx_t[:n_h].tolist())
                train_idx_per_client.append(idx_t[n_h:].tolist() if n > 1 else idx_t.tolist())

            train_loaders = [torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_ds, idx),
                batch_size=min(128, max(1, len(idx))), shuffle=True, drop_last=False)
                for idx in train_idx_per_client]
            full_loaders = [torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_ds, idx),
                batch_size=min(128, max(1, len(idx))), shuffle=True, drop_last=False)
                for idx in client_idx]
            holdout_loaders = [torch.utils.data.DataLoader(
                torch.utils.data.Subset(train_ds, idx),
                batch_size=min(128, max(1, len(idx))), shuffle=False, drop_last=False)
                for idx in holdout_idx_per_client]

            client_models = [copy.deepcopy(global_model).to(device) for _ in range(args.num_clients)]
            for m, L in zip(client_models, train_loaders):
                train_one_client(m, L, args.num_iters, lr=0.01, device=device,
                                 binarize=True, flat=True)

            client_accs = [evaluate(m, test_loader, device, binarize=True,
                                    hard=True, flat=True) for m in client_models]
            results[str(alpha)]['client_mean'].append(float(np.mean(client_accs)))

            # bfs_probe (central labelled probe)
            acc_probe, _ = boolean_feature_selection(
                client_models, probe_loader, test_loader, device,
                num_select=args.num_select, binarize=True, flat=True)
            results[str(alpha)]['bfs_probe'].append(float(acc_probe))

            # bfs_fed: federated standard BFS using each client's training data
            acc_fed, _ = bfs_federated(
                client_models, full_loaders, test_loader, device,
                num_select=args.num_select, binarize=True, flat=True)
            results[str(alpha)]['bfs_fed'].append(float(acc_fed))

            # bfs_fed_holdout: same but uses held-out 20%
            acc_fed_h, _ = bfs_federated(
                client_models, holdout_loaders, test_loader, device,
                num_select=args.num_select, binarize=True, flat=True)
            results[str(alpha)]['bfs_fed_holdout'].append(float(acc_fed_h))

            # bfs_fed_diverse: two-round federated diverse on held-out
            acc_fed_div, _ = bfs_federated_diverse(
                client_models, holdout_loaders, test_loader, device,
                num_select=args.num_select, binarize=True, flat=True,
                diversity_lambda=args.diversity_lambda)
            results[str(alpha)]['bfs_fed_diverse'].append(float(acc_fed_div))

            # diverse_probe: existing probe-based diverse BFS for reference
            acc_div_p, _ = boolean_feature_selection_diverse(
                client_models, probe_loader, test_loader, device,
                num_select=args.num_select, binarize=True, flat=True,
                diversity_lambda=args.diversity_lambda)
            results[str(alpha)]['diverse_probe'].append(float(acc_div_p))

            elapsed = time.time() - t0
            print(f'a={alpha} seed={seed} '
                  f'probe={100*acc_probe:.1f}  fed={100*acc_fed:.1f}  '
                  f'fed_h={100*acc_fed_h:.1f}  fed_div={100*acc_fed_div:.1f}  '
                  f'div_p={100*acc_div_p:.1f}  '
                  f'(took {elapsed:.0f}s, total {(time.time()-t_start)/60:.1f}min)',
                  flush=True)

            # Persist after every (alpha, seed) so partial results survive crashes.
            partial = {a: {m: ({'mean': float(np.mean(v)*100),
                                'std':  float((np.std(v, ddof=1) if len(v) > 1 else 0.0)*100)}
                               if v else None)
                           for m, v in by_m.items()}
                       for a, by_m in results.items()}
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            with open(args.out, 'w') as f:
                json.dump({'config': vars(args), 'raw': results,
                           'partial_summary': partial}, f, indent=2)

    summary = {a: {m: {'mean': float(np.mean(v)*100),
                       'std':  float((np.std(v, ddof=1) if len(v) > 1 else 0.0)*100)}
                   for m, v in by_m.items()}
               for a, by_m in results.items()}
    with open(args.out, 'w') as f:
        json.dump({'config': vars(args), 'raw': results, 'summary': summary}, f, indent=2)
    sentinel = args.out.rsplit('.', 1)[0] + '.DONE'
    with open(sentinel, 'w') as f:
        f.write(f'completed in {(time.time()-t_start)/60:.1f}min\n')

    print(f'\nWrote {args.out}\n')
    cols = ['client_mean'] + method_keys
    header = f'{"alpha":<8}' + ''.join(f'{k:<16}' for k in cols)
    print(header)
    for alpha in args.alphas:
        s = summary[str(alpha)]
        row = f'{alpha:<8}'
        for k in cols:
            row += f'{s[k]["mean"]:5.1f}+/-{s[k]["std"]:.1f}     '
        print(row)


if __name__ == '__main__':
    main()
