"""Approach B1: Shared-wiring FedAvg / FedProx for difflogic networks.

Server initializes one model (fixing wiring), distributes to clients.
Only gate logit tensors are trained locally and communicated.
Supports multi-round training and FedProx regularization.

Usage:
    python federated/local_lgn.py --num_clients 5 --split iid
    python federated/local_lgn.py --num_clients 5 --split iid --num_rounds 5
    python federated/local_lgn.py --num_clients 2 --split noniid --classes_per_client 5 --mu 0.1
"""

import argparse, copy, os, random, sys
os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    get_mnist, get_cifar10, partition_iid, partition_noniid, partition_dirichlet,
    LogicLayer, LogicTreeConv2d, make_difflogic_net, make_conv_difflogic_net,
    evaluate, evaluate_ensemble, evaluate_ensemble_weighted,
    get_gate_weights, set_gate_weights,
    fedavg_gate_weights, fedavg_gate_weights_entropy,
    fedavg_gate_voting, fedavg_gate_ties, fedavg_gate_softconf,
    fedavg_gate_softconf_adaptive, fedavg_gate_adaptive,
    fedavg_gate_funcspace, fedavg_gate_funcspace_conf,
    boolean_feature_selection, boolean_feature_selection_diverse,
    get_client_classes, data_iter, print_gate_stats,
    HAS_TORCHLOGIX,
)
if HAS_TORCHLOGIX:
    from common import make_conv_difflogic_net_tl
from plotting import plot_training_curves


def main():
    parser = argparse.ArgumentParser(description="Approach B1: Shared-wiring FedAvg/FedProx")
    parser.add_argument('--dataset', default='mnist', choices=['mnist', 'cifar10', 'fashion'])
    parser.add_argument('--arch', default='fc', choices=['fc', 'conv'],
                        help="Architecture: 'fc' (flat LogicLayers) or 'conv' (LogicTreeConv2d)")
    parser.add_argument('--size', default='small', choices=['small', 'medium', 'large', 'paper_s', 'paper_b'],
                        help="Conv model size: small/medium/large/paper_s (84K)/paper_b (16M, light param)")
    parser.add_argument('--num_clients', type=int, default=5)
    parser.add_argument('--split', default='iid', choices=['iid', 'noniid', 'dirichlet'])
    parser.add_argument('--alpha', type=float, default=0.5,
                        help="Dirichlet concentration param (lower = more non-IID)")
    parser.add_argument('--classes_per_client', type=int, default=2)
    parser.add_argument('--num_neurons', type=int, default=500)
    parser.add_argument('--num_layers', type=int, default=3)
    parser.add_argument('--tau', type=float, default=10.0)
    parser.add_argument('--gumbel', action='store_true',
                        help="Use Gumbel-ST training (straight-through hard gates)")
    parser.add_argument('--merge', default='fedavg',
                        choices=['fedavg', 'voting', 'ties', 'softconf', 'adaptive'],
                        help="Gate merging method: fedavg, voting, ties, softconf, or adaptive")
    parser.add_argument('--num_iterations', type=int, default=500)
    parser.add_argument('--num_rounds', type=int, default=1,
                        help="Number of federated rounds (1 = single-shot)")
    parser.add_argument('--mu', type=float, default=0.0,
                        help="FedProx proximal term strength (0 = pure FedAvg)")
    parser.add_argument('--local_ce', action='store_true',
                        help="Compute CE only over client's local classes")
    parser.add_argument('--distill_iterations', type=int, default=500)
    parser.add_argument('--kd_temp', type=float, default=4.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--eval_freq', type=int, default=100)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', default='./checkpoints')
    parser.add_argument('--plot', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    algo = f"FedProx (mu={args.mu})" if args.mu > 0 else "FedAvg"
    print(f"Device: {device}")
    print(f"Algorithm: {algo}")

    # ---- Data ----------------------------------------------------------------
    use_conv = (args.arch == 'conv')
    flat = not use_conv  # conv expects 4D input, fc expects flat
    if args.dataset == 'cifar10':
        in_dim = 9216  # 3 * 32 * 32 * 3 thresholds
        binarize = False  # already pre-binarized by transform
        print("\n=== Loading CIFAR-10 ===")
        train_ds, test_ds = get_cifar10(flatten=flat)
    elif args.dataset == 'fashion':
        if use_conv:
            raise ValueError("Conv architecture not wired for Fashion-MNIST")
        from common import get_fashion_mnist
        in_dim = 784
        binarize = True
        print("\n=== Loading Fashion-MNIST ===")
        train_ds, test_ds = get_fashion_mnist()
    else:
        if use_conv:
            raise ValueError("Conv architecture only supported for cifar10")
        in_dim = 784
        binarize = True
        print("\n=== Loading MNIST ===")
        train_ds, test_ds = get_mnist()
    test_bs = 100 if args.size in ('paper_b',) else 1000
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=test_bs, shuffle=False)

    print(f"\n=== Partitioning data ({args.split}) ===")
    if args.split == 'iid':
        client_idx = partition_iid(train_ds, args.num_clients)
    elif args.split == 'dirichlet':
        client_idx = partition_dirichlet(train_ds, args.num_clients, alpha=args.alpha)
    else:
        client_idx = partition_noniid(train_ds, args.num_clients, args.classes_per_client)

    # ---- Server initializes global model (wiring fixed once) -----------------
    print(f"\n=== Server initializes global difflogic model ===")
    if use_conv:
        use_tl = HAS_TORCHLOGIX and device.type == 'cuda'
        backend = "torchlogix" if use_tl else "native"
        print(f"  Architecture: conv difflogic (depth=2, size={args.size}, tau={args.tau}, backend={backend})")
        if use_tl:
            global_model = make_conv_difflogic_net_tl(classes=10, tau=args.tau, size=args.size, device=str(device))
        else:
            global_model = make_conv_difflogic_net(classes=10, tau=args.tau, size=args.size).to(device)
        num_gates = sum(p.shape[0] if p.dim() == 2 else p.shape[1] * p.shape[0]
                        for p in global_model.parameters() if p.shape[-1] == 16)
    else:
        print(f"  Architecture: {in_dim} -> {args.num_neurons}x{args.num_layers} -> GroupSum(10, tau={args.tau})")
        global_model = make_difflogic_net(in_dim, args.num_neurons, args.num_layers, 10,
                                          tau=args.tau, gumbel=args.gumbel).to(device)
        num_gates = sum(1 for m in global_model.modules() if isinstance(m, LogicLayer)) * args.num_neurons
    print(f"  Gates: {num_gates}, params: {sum(p.numel() for p in global_model.parameters())}")

    # Distribute to clients (same wiring, same initial logits)
    client_models = [copy.deepcopy(global_model) for _ in range(args.num_clients)]

    client_loaders = []
    for i in range(args.num_clients):
        subset = torch.utils.data.Subset(train_ds, client_idx[i])
        bs = min(args.batch_size, max(1, len(client_idx[i])))
        drop_last = len(client_idx[i]) >= 2 * args.batch_size
        loader = torch.utils.data.DataLoader(subset, batch_size=bs, shuffle=True, drop_last=drop_last)
        client_loaders.append(loader)

    # ---- Federated rounds ----------------------------------------------------
    iters_per_round = args.num_iterations // args.num_rounds
    print(f"\n=== Federated training: {args.num_rounds} round(s), "
          f"{iters_per_round} iters/client/round ===")

    client_histories = [[] for _ in range(args.num_clients)]
    round_boundaries = []

    for rnd in range(1, args.num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"  ROUND {rnd}/{args.num_rounds}")
        print(f"{'='*60}")

        # Save global weights for FedProx proximal term
        global_params = [p.data.clone() for p in global_model.parameters()]

        # ---- Local training --------------------------------------------------
        cum_offset = (rnd - 1) * iters_per_round
        if rnd > 1:
            round_boundaries.append(cum_offset)

        for i in range(args.num_clients):
            local_classes, label_remap = get_client_classes(train_ds, client_idx[i])
            label_remap = label_remap.to(device)
            if args.local_ce:
                print(f"\n--- Client {i} ({len(client_idx[i])} samples, local CE over {local_classes}) ---")
            else:
                print(f"\n--- Client {i} ({len(client_idx[i])} samples) ---")
            model = client_models[i]
            if args.size in ('paper_s', 'paper_b'):
                optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.002)
            else:
                optimizer = optim.Adam(model.parameters(), lr=args.lr)
            gen = data_iter(client_loaders[i])

            model.train()
            for it in range(1, iters_per_round + 1):
                x, y = next(gen)
                x, y = x.to(device), y.to(device)
                if flat:
                    x_in = x.view(x.size(0), -1)
                else:
                    x_in = x
                if binarize:
                    x_in = (x_in > 0.5).float()

                logits = model(x_in)
                if args.local_ce:
                    loss = F.cross_entropy(logits[:, local_classes], label_remap[y])
                else:
                    loss = F.cross_entropy(logits, y)

                # FedProx proximal term
                if args.mu > 0:
                    prox = sum(((p - pg) ** 2).sum()
                               for p, pg in zip(model.parameters(), global_params))
                    loss = loss + (args.mu / 2) * prox

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if it % args.eval_freq == 0 or it == iters_per_round:
                    torch.cuda.empty_cache() if device.type == 'cuda' else None
                    soft_acc = evaluate(model, test_loader, device, binarize=binarize, hard=False, flat=flat)
                    hard_acc = evaluate(model, test_loader, device, binarize=binarize, hard=True, flat=flat)
                    print(f"  Iter {it}/{iters_per_round} - Loss: {loss.item():.3f} - "
                          f"Soft: {100*soft_acc:.1f}%, Hard: {100*hard_acc:.1f}%")
                    client_histories[i].append((cum_offset + it, loss.item(), soft_acc, hard_acc))
                    model.train()

        # Save checkpoints BEFORE merge (in case merge OOMs)
        save_path = os.path.join(args.save_dir, f'b1_{args.dataset}_{args.split}_K{args.num_clients}')
        os.makedirs(save_path, exist_ok=True)
        for i, m in enumerate(client_models):
            torch.save(m, os.path.join(save_path, f'client_{i}.pt'))
        torch.save(global_model, os.path.join(save_path, 'global.pt'))
        print(f"  Checkpoints saved to {save_path}")

        # Move all client models to CPU to free GPU memory for merge evaluation
        # For large models, free memory before merge evaluation
        if args.size in ('paper_b',) and device.type == 'cuda':
            torch.cuda.empty_cache()
            print("  (Cleared CUDA cache before merge evaluation)")

        # ---- Primary merge method ------------------------------------------------
        merge_name = args.merge.upper()
        print(f"\n--- {merge_name}: aggregate gate logits ---")
        if args.merge == 'voting':
            avg_weights = fedavg_gate_voting(client_models)
        elif args.merge == 'ties':
            avg_weights = fedavg_gate_ties(client_models)
        elif args.merge == 'softconf':
            avg_weights = fedavg_gate_softconf(client_models)
        elif args.merge == 'adaptive':
            avg_weights = fedavg_gate_adaptive(client_models)
        else:
            avg_weights = fedavg_gate_weights(client_models)
        set_gate_weights(global_model, avg_weights)

        fedavg_soft = evaluate(global_model, test_loader, device, binarize=binarize, hard=False, flat=flat)
        fedavg_hard = evaluate(global_model, test_loader, device, binarize=binarize, hard=True, flat=flat)
        print(f"  Global model — Soft: {100*fedavg_soft:.1f}%, Hard: {100*fedavg_hard:.1f}%")

        # ---- Also evaluate other merge methods for comparison ----------------
        merge_results = {}
        for merge_method, merge_fn, merge_label in [
            ('fedavg', fedavg_gate_weights, 'FedAvg'),
            ('entropy', fedavg_gate_weights_entropy, 'Entropy-wt FedAvg'),
            ('voting', fedavg_gate_voting, 'Voting'),
            ('ties', fedavg_gate_ties, 'TIES'),
            ('softconf', fedavg_gate_softconf, 'SoftConf'),
            ('softconf_adaptive', fedavg_gate_softconf_adaptive, 'SoftConf-A'),
            ('adaptive', fedavg_gate_adaptive, 'Adaptive'),
            ('funcspace', fedavg_gate_funcspace, 'FuncSpace'),
            ('funcspace_conf', fedavg_gate_funcspace_conf, 'FuncSpace-Conf'),
        ]:
            if merge_method == args.merge:
                merge_results[merge_label] = (fedavg_soft, fedavg_hard)
                continue
            w = merge_fn(client_models)
            m_tmp = copy.deepcopy(global_model)
            set_gate_weights(m_tmp, w)
            s = evaluate(m_tmp, test_loader, device, binarize=binarize, hard=False, flat=flat)
            h = evaluate(m_tmp, test_loader, device, binarize=binarize, hard=True, flat=flat)
            merge_results[merge_label] = (s, h)
            print(f"  {merge_label} — Soft: {100*s:.1f}%, Hard: {100*h:.1f}%")

        # Legacy names for summary
        fedavg_ent_soft = merge_results.get('Entropy-wt FedAvg', (0, 0))[0]
        fedavg_ent_hard = merge_results.get('Entropy-wt FedAvg', (0, 0))[1]

        # ---- Broadcast back to clients ---------------------------------------
        if rnd < args.num_rounds:
            for m in client_models:
                set_gate_weights(m, avg_weights)
            print(f"  Broadcast global weights to {args.num_clients} clients")

    # ---- Final client accuracies ---------------------------------------------
    print("\n=== Final individual client accuracies ===")
    client_accs_soft, client_accs_hard = [], []
    for i in range(args.num_clients):
        s = evaluate(client_models[i], test_loader, device, binarize=binarize, hard=False, flat=flat)
        h = evaluate(client_models[i], test_loader, device, binarize=binarize, hard=True, flat=flat)
        client_accs_soft.append(s)
        client_accs_hard.append(h)
        print(f"  Client {i}: Soft {100*s:.1f}%, Hard {100*h:.1f}%")

    # ---- Ensemble ------------------------------------------------------------
    print("\n=== Ensemble (average output logits) ===")
    ens_soft = evaluate_ensemble(client_models, test_loader, device, binarize=binarize, hard=False, flat=flat)
    ens_hard = evaluate_ensemble(client_models, test_loader, device, binarize=binarize, hard=True, flat=flat)
    print(f"  Ensemble soft:  {100*ens_soft:.1f}%")
    print(f"  Ensemble hard:  {100*ens_hard:.1f}%")

    print("\n=== Ensemble (confidence-weighted) ===")
    ensw_soft = evaluate_ensemble_weighted(client_models, test_loader, device, binarize=binarize, hard=False, flat=flat)
    ensw_hard = evaluate_ensemble_weighted(client_models, test_loader, device, binarize=binarize, hard=True, flat=flat)
    print(f"  Weighted ensemble soft:  {100*ensw_soft:.1f}%")
    print(f"  Weighted ensemble hard:  {100*ensw_hard:.1f}%")

    # ---- Boolean feature selection -------------------------------------------
    print("\n=== Boolean feature selection ===")
    # Subsample probe for large models to avoid OOM
    if args.size in ('paper_b',):
        probe_n = min(5000, len(train_ds))
        probe_idx = torch.randperm(len(train_ds))[:probe_n].tolist()
        probe_subset = torch.utils.data.Subset(train_ds, probe_idx)
        probe_loader = torch.utils.data.DataLoader(probe_subset, batch_size=500, shuffle=False)
        print(f"  Using {probe_n} probe samples (subsampled for large model)")
    else:
        probe_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=1000, shuffle=False)
    bfs_results = {}
    for ns in [50, 100]:
        acc, sel = boolean_feature_selection(
            client_models, probe_loader, test_loader, device, num_select=ns,
            binarize=binarize, flat=flat)
        bfs_results[ns] = acc
        # Show gate provenance
        K = args.num_clients
        # Find last logic layer's output dim = number of gates per model in BFS
        logic_params = [p for p in client_models[0].parameters() if p.shape[-1] == 16]
        gates_per_model = logic_params[-1].shape[0] if logic_params[-1].dim() == 2 else logic_params[-1].shape[0] * logic_params[-1].shape[1]
        provenance = [0] * K
        for c in range(10):
            for idx in sel[c]:
                provenance[idx.item() // gates_per_model] += 1
        prov_str = ", ".join(f"client{i}={provenance[i]}" for i in range(K))
        print(f"  num_select={ns}: {100*acc:.1f}%  ({prov_str})")

    # ---- Diverse BFS (Fourier-aware) -----------------------------------------
    print("\n=== Diverse BFS (Fourier-aware) ===")
    for ns, lam in [(100, 0.3), (100, 0.5), (100, 1.0)]:
        acc_d, _ = boolean_feature_selection_diverse(
            client_models, probe_loader, test_loader, device, num_select=ns,
            binarize=binarize, flat=flat, diversity_lambda=lam)
        bfs_results[f'div_{ns}_l{lam}'] = acc_d
        print(f"  diverse n={ns}, lambda={lam}: {100*acc_d:.1f}%")

    # ---- Ensemble distillation -----------------------------------------------
    distill_history = []
    distill_soft = distill_hard = best_distill_hard = 0.0
    if args.distill_iterations > 0:
        print(f"\n=== Ensemble distillation ({args.distill_iterations} iterations) ===")
        full_train_loader = torch.utils.data.DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True)

        # Keep the residual initialization from the underlying conv difflogic
        # network (w_3=5 pass-through bias).  An earlier version of this code
        # called nn.init.normal_ on every 16D parameter to "freshly" initialize
        # the student; that overwrites the residual init and causes vanishing
        # gradients on deep conv-difflogic stacks (verified empirically on
        # paper_s: distillation collapses to 10% with the random reinit and
        # converges normally without it).  MNIST FC tolerates the random init
        # because the network is only 3 layers deep, but the reinit is wrong
        # in general and we rely on the same starting point the trained
        # clients used.
        student = copy.deepcopy(global_model)

        optimizer = optim.Adam(student.parameters(), lr=args.lr)
        gen = data_iter(full_train_loader)
        T = args.kd_temp

        for m in client_models:
            m.eval()

        student.train()
        for it in range(1, args.distill_iterations + 1):
            x, y = next(gen)
            x, y = x.to(device), y.to(device)
            if flat:
                x_in = x.view(x.size(0), -1)
            else:
                x_in = x
            if binarize:
                x_in = (x_in > 0.5).float()

            student_logits = student(x_in)
            ce_loss = F.cross_entropy(student_logits, y)

            with torch.no_grad():
                teacher_logits = torch.stack([m(x_in) for m in client_models]).mean(0)
                teacher_soft = F.softmax(teacher_logits / T, dim=-1)

            kd_loss = F.kl_div(F.log_softmax(student_logits / T, dim=-1),
                               teacher_soft, reduction='batchmean') * (T ** 2)
            loss = kd_loss + ce_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if it % args.eval_freq == 0:
                sa = evaluate(student, test_loader, device, binarize=binarize, hard=False, flat=flat)
                ha = evaluate(student, test_loader, device, binarize=binarize, hard=True, flat=flat)
                best_distill_hard = max(best_distill_hard, ha)
                print(f"  Iter {it}/{args.distill_iterations} - Loss: {loss.item():.3f} "
                      f"(KD:{kd_loss.item():.3f} CE:{ce_loss.item():.3f}) - "
                      f"Soft: {100*sa:.1f}%, Hard: {100*ha:.1f}%")
                distill_history.append((it, loss.item(), kd_loss.item(), ce_loss.item(), sa, ha))
                student.train()

        distill_soft = evaluate(student, test_loader, device, binarize=binarize, hard=False, flat=flat)
        distill_hard = evaluate(student, test_loader, device, binarize=binarize, hard=True, flat=flat)
        best_distill_hard = max(best_distill_hard, distill_hard)

    # ---- Gate stats ----------------------------------------------------------
    print_gate_stats(global_model, label=f"{algo} global")

    # ---- Plot curves ---------------------------------------------------------
    if args.plot:
        fig_path = f'./federated/figures/b1_{args.dataset}_{args.split}_K{args.num_clients}_curves.png'
        plot_training_curves(
            client_histories, fig_path,
            title_prefix="B1: ",
            distill_history=distill_history if distill_history else None,
            round_boundaries=round_boundaries if round_boundaries else None,
        )

    # ---- Save ----------------------------------------------------------------
    save_path = os.path.join(args.save_dir, f'b1_{args.dataset}_{args.split}_K{args.num_clients}')
    os.makedirs(save_path, exist_ok=True)
    for i, m in enumerate(client_models):
        torch.save(m, os.path.join(save_path, f'client_{i}.pt'))
    torch.save(global_model, os.path.join(save_path, 'global.pt'))

    # ---- Summary -------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  SUMMARY — B1: Shared Wiring + {algo}")
    print(f"{'='*60}")
    print(f"  Dataset: {args.dataset}, Split: {args.split}, Clients: {args.num_clients}, Rounds: {args.num_rounds}")
    if args.split == 'noniid':
        print(f"  Classes per client: {args.classes_per_client}")
    if args.split == 'dirichlet':
        print(f"  Dirichlet alpha: {args.alpha}")
    if use_conv:
        print(f"  Architecture: conv difflogic (depth=2, tau={args.tau})")
    else:
        print(f"  Architecture: {in_dim} -> {args.num_neurons}x{args.num_layers} -> 10 (tau={args.tau})")
    print(f"  Training: {args.num_iterations} total iters/client ({iters_per_round}/round), lr={args.lr}")
    print(f"  Merge: {args.merge}, Gumbel-ST: {args.gumbel}")
    if args.mu > 0:
        print(f"  FedProx mu: {args.mu}")
    print()
    print(f"  {'Method':<30s} {'Soft':>8s}  {'Hard':>8s}")
    print(f"  {'-'*30} {'-'*8}  {'-'*8}")
    for i in range(args.num_clients):
        print(f"  {'Client ' + str(i):<30s} {100*client_accs_soft[i]:>7.1f}%  {100*client_accs_hard[i]:>7.1f}%")
    mean_soft = 100 * np.mean(client_accs_soft)
    mean_hard = 100 * np.mean(client_accs_hard)
    print(f"  {'Client mean':<30s} {mean_soft:>7.1f}%  {mean_hard:>7.1f}%")
    for label, (s, h) in merge_results.items():
        print(f"  {label + ' (global)':<30s} {100*s:>7.1f}%  {100*h:>7.1f}%")
    print(f"  {'Ensemble (avg output)':<30s} {100*ens_soft:>7.1f}%  {100*ens_hard:>7.1f}%")
    print(f"  {'Ensemble (confidence-wt)':<30s} {100*ensw_soft:>7.1f}%  {100*ensw_hard:>7.1f}%")
    for ns, acc in bfs_results.items():
        print(f"  {f'Bool feature sel (n={ns})':<30s} {'':>8s}  {100*acc:>7.1f}%")
    if args.distill_iterations > 0:
        print(f"  {'Distilled student':<30s} {100*distill_soft:>7.1f}%  {100*distill_hard:>7.1f}%")
        print(f"  {'Distilled best hard':<30s} {'':>8s}  {100*best_distill_hard:>7.1f}%")
    print(f"\n  Models saved to {save_path}")


if __name__ == '__main__':
    main()
