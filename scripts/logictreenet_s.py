"""Multi-seed diverse-BFS at alpha=0.5 on Paper-S (CIFAR-10 conv-LGN).

Trains K=5 Paper-S clients per seed on a Dirichlet partition (alpha=0.5,
10K iters each), then evaluates standard BFS and diverse BFS (lambda=0.3)
on the same trained clients.  Reports mean +/- std across seeds.

Designed for a single RTX 5090 (32 GB).  Paper-S = 84K gates, ~1.3M
params, ~2 GB per model; we hold one client at a time during training
and all five during BFS evaluation.

Output: scripts/results/multiseed_paper_s_diverse_bfs.json
Done sentinel: scripts/results/multiseed_paper_s_diverse_bfs.DONE
"""
import argparse, copy, json, os, random, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'federated'))
from common import (
    make_conv_difflogic_net_tl, get_cifar10, partition_dirichlet, evaluate,
    train_one_client,
    boolean_feature_selection, boolean_feature_selection_diverse,
    evaluate_ensemble, fedavg_gate_softconf, fedavg_gate_weights,
    set_gate_weights,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alpha', type=float, default=0.5)
    ap.add_argument('--seeds', nargs='+', type=int, default=[42, 43, 44])
    ap.add_argument('--num_clients', type=int, default=5)
    ap.add_argument('--num_iters', type=int, default=10000)
    ap.add_argument('--num_select', type=int, default=100)
    ap.add_argument('--diversity_lambda', type=float, default=0.3)
    ap.add_argument('--lr', type=float, default=0.02)
    ap.add_argument('--probe_size', type=int, default=2000,
                    help='Subsample of train set used as BFS probe (memory cap).')
    ap.add_argument('--out', default='scripts/results/multiseed_paper_s_diverse_bfs.json')
    ap.add_argument('--save_dir', default=None,
                    help="If set, save trained client models to "
                         "<save_dir>/seed_<S>/client_<K>.pt for later re-evaluation.")
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}; CUDA: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print(f'GPU: {torch.cuda.get_device_name()}, '
              f'{torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')

    train_ds, test_ds = get_cifar10(flatten=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=200, shuffle=False)

    probe_idx = torch.randperm(len(train_ds), generator=torch.Generator().manual_seed(0))[:args.probe_size].tolist()
    probe_ds = torch.utils.data.Subset(train_ds, probe_idx)
    probe_loader = torch.utils.data.DataLoader(probe_ds, batch_size=200, shuffle=False)

    results = {
        'client_mean': [],
        'fedavg': [],
        'softconf': [],
        'ensemble': [],
        'bfs': [],
        'diverse_bfs': [],
    }
    t_start = time.time()

    for seed in args.seeds:
        t0 = time.time()
        torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)

        # Build a fresh global model under this seed; clients deepcopy from it.
        global_model = make_conv_difflogic_net_tl(
            classes=10, depth=2, size='paper_s', device=device.type)

        client_idx = partition_dirichlet(train_ds, args.num_clients, alpha=args.alpha)
        for k in range(args.num_clients):
            if len(client_idx[k]) == 0:
                donor = max(range(args.num_clients), key=lambda i: len(client_idx[i]))
                client_idx[k] = [client_idx[donor][0]]

        loaders = [torch.utils.data.DataLoader(
            torch.utils.data.Subset(train_ds, idx),
            batch_size=min(128, len(idx)), shuffle=True, drop_last=False)
            for idx in client_idx]

        client_models = []
        for k in range(args.num_clients):
            m = copy.deepcopy(global_model).to(device)
            t_c = time.time()
            train_one_client(m, loaders[k], args.num_iters, lr=args.lr, device=device,
                             binarize=False, flat=False, optimizer_kind='adamw',
                             weight_decay=0.002)
            client_models.append(m)
            print(f'  seed={seed} client={k} trained in {time.time()-t_c:.0f}s '
                  f'({len(client_idx[k])} samples)', flush=True)
            if args.save_dir:
                ck = os.path.join(args.save_dir, f'seed_{seed}', f'client_{k}.pt')
                os.makedirs(os.path.dirname(ck), exist_ok=True)
                torch.save({'state_dict': m.state_dict(),
                            'sample_idx': client_idx[k],
                            'config': vars(args)}, ck)

        client_accs = [evaluate(m, test_loader, device, binarize=False, hard=True, flat=False)
                       for m in client_models]
        results['client_mean'].append(float(np.mean(client_accs)))
        print(f'  seed={seed} client_mean={100*np.mean(client_accs):.1f}%', flush=True)

        merged = copy.deepcopy(global_model).to(device)
        set_gate_weights(merged, fedavg_gate_weights(client_models))
        acc_fedavg = evaluate(
            merged, test_loader, device, binarize=False, hard=True, flat=False)
        results['fedavg'].append(float(acc_fedavg))
        del merged

        merged = copy.deepcopy(global_model).to(device)
        set_gate_weights(merged, fedavg_gate_softconf(client_models))
        acc_softconf = evaluate(
            merged, test_loader, device, binarize=False, hard=True, flat=False)
        results['softconf'].append(float(acc_softconf))
        del merged

        acc_ensemble = evaluate_ensemble(
            client_models, test_loader, device, binarize=False, hard=True, flat=False)
        results['ensemble'].append(float(acc_ensemble))

        acc_bfs, _ = boolean_feature_selection(
            client_models, probe_loader, test_loader, device,
            num_select=args.num_select, binarize=False, flat=False)
        results['bfs'].append(float(acc_bfs))

        acc_div, _ = boolean_feature_selection_diverse(
            client_models, probe_loader, test_loader, device,
            num_select=args.num_select, binarize=False, flat=False,
            diversity_lambda=args.diversity_lambda)
        results['diverse_bfs'].append(float(acc_div))

        elapsed = time.time() - t0
        print(f'alpha={args.alpha} seed={seed} '
              f'FedAvg={100*acc_fedavg:.1f}%  SoftConf={100*acc_softconf:.1f}%  '
              f'Ensemble={100*acc_ensemble:.1f}%  BFS={100*acc_bfs:.1f}%  '
              f'Diverse={100*acc_div:.1f}%  '
              f'(took {elapsed/60:.1f}min, total {(time.time()-t_start)/60:.1f}min)',
              flush=True)

        # Persist running JSON each seed (so partial results survive crashes).
        partial = {'config': vars(args), 'raw': results,
                   'partial_summary': {k: {'mean': float(np.mean(v) * 100),
                                           'std':  float((np.std(v, ddof=1) if len(v) > 1 else 0.0) * 100)}
                                       for k, v in results.items() if v}}
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, 'w') as f:
            json.dump(partial, f, indent=2)

        # Free GPU memory between seeds.
        for m in client_models: m.cpu()
        del client_models, global_model
        torch.cuda.empty_cache()

    summary = {k: {'mean': float(np.mean(v) * 100),
                   'std':  float((np.std(v, ddof=1) if len(v) > 1 else 0.0) * 100)}
               for k, v in results.items()}
    with open(args.out, 'w') as f:
        json.dump({'config': vars(args), 'raw': results, 'summary': summary}, f, indent=2)

    # Done sentinel for the orchestrator.
    sentinel = args.out.rsplit('.', 1)[0] + '.DONE'
    with open(sentinel, 'w') as f:
        f.write(f'completed in {(time.time()-t_start)/60:.1f}min\n')

    print(f'\nWrote {args.out}\n')
    print(f'{"method":<14}{"acc (mean+/-std)":<20}')
    for k in ['client_mean', 'fedavg', 'softconf', 'ensemble', 'bfs', 'diverse_bfs']:
        s = summary[k]
        print(f'{k:<14}{s["mean"]:.1f}+/-{s["std"]:.1f}')


if __name__ == '__main__':
    main()
