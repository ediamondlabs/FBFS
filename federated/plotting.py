"""Training curve plotting for federated difflogic experiments."""

import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

COLORS = ['#4c72b0', '#dd8452', '#55a868', '#c44e52', '#8172b3',
          '#937860', '#da8bc3', '#8c8c8c', '#ccb974', '#64b5cd']


def plot_training_curves(client_histories, save_path, title_prefix="",
                         distill_history=None, round_boundaries=None):
    """Plot training loss and accuracy curves.

    Args:
        client_histories: list of lists, one per client. Each inner list has
            tuples (iteration, loss, soft_acc, hard_acc).
        save_path: file path for the saved figure.
        title_prefix: string prepended to panel titles.
        distill_history: optional list of (iter, total_loss, kd_loss, ce_loss,
            soft_acc, hard_acc) for distillation curves.
        round_boundaries: optional list of iteration numbers where federated
            rounds start (for vertical dashed lines).
    """
    has_clients = bool(client_histories)
    has_distill = bool(distill_history)
    n_panels = (2 if has_clients else 0) + (1 if has_distill else 0)
    if n_panels == 0:
        return

    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4))
    if n_panels == 1:
        axes = [axes]

    pi = 0  # panel index

    if has_clients:
        ax_loss, ax_acc = axes[pi], axes[pi + 1]
        pi += 2

        # Left panel: training loss
        for ci, hist in enumerate(client_histories):
            iters = [h[0] for h in hist]
            losses = [h[1] for h in hist]
            color = COLORS[ci % len(COLORS)]
            ax_loss.plot(iters, losses, color=color, label=f'Client {ci}', alpha=0.8)

        if round_boundaries:
            for rb in round_boundaries:
                ax_loss.axvline(rb, color='gray', linestyle='--', alpha=0.4)

        ax_loss.set_xlabel('Iteration')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title(f'{title_prefix}Training Loss' if title_prefix else 'Training Loss')
        ax_loss.legend(fontsize=7)
        ax_loss.grid(alpha=0.3)

        # Right panel: test accuracy
        for ci, hist in enumerate(client_histories):
            iters = [h[0] for h in hist]
            soft = [100 * h[2] for h in hist]
            hard = [100 * h[3] for h in hist]
            color = COLORS[ci % len(COLORS)]
            ax_acc.plot(iters, soft, color=color, linestyle='-', alpha=0.8,
                        label=f'Client {ci} soft')
            ax_acc.plot(iters, hard, color=color, linestyle='--', alpha=0.6,
                        label=f'Client {ci} hard')

        if round_boundaries:
            for rb in round_boundaries:
                ax_acc.axvline(rb, color='gray', linestyle='--', alpha=0.4)

        ax_acc.set_xlabel('Iteration')
        ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.set_title(f'{title_prefix}Test Accuracy' if title_prefix else 'Test Accuracy')
        ax_acc.legend(fontsize=6)
        ax_acc.grid(alpha=0.3)

    # Distillation panel (optional)
    if has_distill:
        ax_d = axes[pi]
        iters = [h[0] for h in distill_history]
        total = [h[1] for h in distill_history]
        kd = [h[2] for h in distill_history]
        ce = [h[3] for h in distill_history]
        soft = [100 * h[4] for h in distill_history]
        hard = [100 * h[5] for h in distill_history]

        ax_d.plot(iters, total, color=COLORS[0], label='Total loss')
        ax_d.plot(iters, kd, color=COLORS[1], linestyle='--', label='KD loss')
        ax_d.plot(iters, ce, color=COLORS[2], linestyle='--', label='CE loss')
        ax_d.set_xlabel('Iteration')
        ax_d.set_ylabel('Loss')
        ax_d.legend(fontsize=7, loc='upper left')
        ax_d.grid(alpha=0.3)

        ax_d2 = ax_d.twinx()
        ax_d2.plot(iters, soft, color=COLORS[3], linestyle='-', alpha=0.8, label='Soft acc')
        ax_d2.plot(iters, hard, color=COLORS[4], linestyle='--', alpha=0.6, label='Hard acc')
        ax_d2.set_ylabel('Accuracy (%)')
        ax_d2.legend(fontsize=7, loc='lower right')

        ax_d.set_title(f'{title_prefix}Distillation' if title_prefix else 'Distillation')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"\n  Training curves saved to {save_path}")
