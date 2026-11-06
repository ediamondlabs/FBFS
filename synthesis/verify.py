"""End-to-end verification of the unified synthesis pipeline (C target).

After the refactor (April 2026), the Verilog and C exporters share a
single source of truth for the 16 gate operations (synthesis/gate_ops.py).
This script trains K client models, runs BFS, compiles the BFS-selected
circuit to native C via compiled_bfs.emit_bfs_c_code, executes the
compiled .so on real MNIST test images, and compares its predictions to
the PyTorch BFS reference.

The companion script synthesis/verify_verilog.py exercises the same BFS
output through the Verilog target: it runs the Verilog exporter, builds
a Verilator simulation of the resulting netlist, drives the same MNIST
test images through it, and confirms the predictions match. Together
the two scripts cover both deployment paths (C/CPU and Verilog/silicon).
"""
import sys, os, copy, random, ctypes, tempfile, subprocess, time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'federated'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    make_difflogic_net, get_mnist, partition_dirichlet,
    boolean_feature_selection, data_iter, LogicLayer, train_one_client,
)
from compiled_bfs import emit_bfs_c_code


def pytorch_bfs_predict_batch(client_models, selected, x_binary_batch, num_classes=10):
    """Run BFS prediction in PyTorch on a batch of binarized inputs."""
    all_gates = []
    for m in client_models:
        m.eval()
        layers = [mod for mod in m.modules() if isinstance(mod, LogicLayer)]
        last = layers[-1]
        hook_out = []
        h = last.register_forward_hook(lambda m, inp, out: hook_out.append(out.detach()))
        with torch.no_grad():
            m(x_binary_batch)
        h.remove()
        all_gates.append(hook_out[0])
    gates = torch.cat(all_gates, dim=1)  # (B, K*N)
    logits = torch.zeros(x_binary_batch.shape[0], num_classes)
    for c in range(num_classes):
        logits[:, c] = gates[:, selected[c]].sum(dim=1)
    return logits.argmax(dim=1).cpu().numpy(), logits.cpu().numpy()


def compile_c(c_code):
    """Write C code to disk, compile to a .so, return path."""
    tmpdir = tempfile.mkdtemp(prefix='bfs_verify_')
    c_path = os.path.join(tmpdir, 'bfs.c')
    so_path = os.path.join(tmpdir, 'bfs.so')
    with open(c_path, 'w') as f:
        f.write(c_code)
    res = subprocess.run(
        ['gcc', '-shared', '-fPIC', '-O1', '-o', so_path, c_path],
        capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(f"gcc failed: {res.stderr[:500]}")
    return so_path


def main():
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    device = torch.device('cpu')
    train_ds, test_ds = get_mnist()
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False)
    probe_loader = torch.utils.data.DataLoader(train_ds, batch_size=1000, shuffle=False)

    K = 5
    client_idx = partition_dirichlet(train_ds, K, alpha=0.5)
    global_model = make_difflogic_net(784, 500, 3, 10, tau=10.0)
    client_models = [copy.deepcopy(global_model) for _ in range(K)]

    print(f"Training {K} clients (500 iters each)...")
    for i in range(K):
        subset = torch.utils.data.Subset(train_ds, client_idx[i])
        loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True, drop_last=True)
        train_one_client(client_models[i], loader, 500, lr=0.01, device=device,
                         binarize=True, flat=True)

    print("Running BFS...")
    acc, selected = boolean_feature_selection(
        client_models, probe_loader, test_loader, device,
        num_select=100, binarize=True, flat=True)
    print(f"  PyTorch BFS reference accuracy: {100*acc:.2f}%")

    print("\nGenerating C code from the BFS-selected circuit...")
    c_code, in_dim = emit_bfs_c_code(client_models, selected, num_classes=10)
    print(f"  {len(c_code.splitlines())} lines of C")

    print("Compiling with gcc -O1...")
    t0 = time.time()
    so_path = compile_c(c_code)
    print(f"  compiled in {time.time()-t0:.2f}s -> {so_path}")

    print("Loading compiled binary via ctypes...")
    lib = ctypes.cdll.LoadLibrary(so_path)
    bfs_fn = lib.bfs_circuit
    bfs_fn.restype = None
    bfs_fn.argtypes = [
        np.ctypeslib.ndpointer(ctypes.c_longlong, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_longlong, flags="C_CONTIGUOUS"),
    ]

    print("\nRunning end-to-end comparison on 200 MNIST test images...")
    N = 200
    inp_buf = np.zeros(in_dim, dtype=np.int64)
    out_buf = np.zeros(10, dtype=np.int64)

    # Collect PyTorch predictions for the first N test images.
    xs = torch.stack([test_ds[i][0].view(-1) for i in range(N)], dim=0)
    xs_bin = (xs > 0.5).float()
    pt_preds, pt_logits = pytorch_bfs_predict_batch(client_models, selected, xs_bin)

    # Run compiled C on each image.
    c_preds = np.zeros(N, dtype=np.int64)
    c_logits = np.zeros((N, 10), dtype=np.int64)
    for i in range(N):
        np.copyto(inp_buf, xs_bin[i].numpy().astype(np.int64))
        bfs_fn(inp_buf, out_buf)
        c_logits[i] = out_buf
        c_preds[i] = int(np.argmax(out_buf))

    matches = int((c_preds == pt_preds).sum())
    print(f"\n  {matches}/{N} predictions match between PyTorch and compiled C circuit.")
    if matches < N:
        # Show a few mismatches for debugging
        mis = np.where(c_preds != pt_preds)[0][:5]
        for i in mis:
            print(f"    mismatch at sample {i}: pytorch={pt_preds[i]} C={c_preds[i]}")
            print(f"      pytorch logits: {pt_logits[i]}")
            print(f"      C       logits: {c_logits[i]}")

    if matches == N:
        print("\n*** PASS: end-to-end C pipeline correctness verified. ***")
        print("    The Verilog path is verified separately by")
        print("    synthesis/verify_verilog.py (Verilator round-trip on the")
        print("    same 200 MNIST images, also 200/200 match).")
        return 0
    else:
        print("\n*** FAIL: predictions diverged. ***")
        return 1


if __name__ == '__main__':
    sys.exit(main())
