"""End-to-end Verilog verification of the BFS-selected circuit.

Mirror of synthesis/verify.py but for the Verilog deployment path:

    1. Train K=5 client models (same seed/hyperparameters as verify.py).
    2. Run BFS, get `selected`.
    3. Export the BFS-selected circuit to Verilog (with the argmax
       comparator wired up so `prediction` is a real one-hot output).
    4. Verilate (`verilator --cc --exe`) and compile a tiny C++ harness
       that drives N MNIST test images through the simulated circuit.
    5. Compare the Verilator predictions against the PyTorch BFS
       reference. Report N/N match and PASS/FAIL.

Single source of truth for gate semantics still lives in
`synthesis/gate_ops.py`; this script just confirms that the Verilog
emitter wires the resulting expressions into a top-level module that
Verilator simulates correctly.
"""
import sys, os, copy, random, tempfile, subprocess, time, shutil
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
from export_verilog import export_bfs_circuit, run_yosys_stats


HARNESS_CPP = r"""
// Verilator harness for the BFS-selected MNIST circuit.
// Reads N (in_dim*N bytes) of {0,1} input bits from stdin, drives the
// simulated `bfs_circuit` once per sample, and writes the one-hot
// 10-bit prediction (one byte == argmax index) to stdout.
#include "Vbfs_circuit.h"
#include "verilated.h"
#include <cstdio>
#include <cstdint>
#include <vector>

static constexpr int IN_DIM = __IN_DIM__;
static constexpr int N_SAMPLES = __N_SAMPLES__;
static constexpr int NUM_CLASSES = 10;

int main(int argc, char** argv) {
    Verilated::commandArgs(argc, argv);
    Vbfs_circuit* dut = new Vbfs_circuit;

    std::vector<uint8_t> bits(IN_DIM);
    std::vector<uint8_t> preds(N_SAMPLES);

    for (int s = 0; s < N_SAMPLES; ++s) {
        if (fread(bits.data(), 1, IN_DIM, stdin) != (size_t)IN_DIM) {
            fprintf(stderr, "harness: short read at sample %d\n", s);
            delete dut;
            return 2;
        }
        // Pack the IN_DIM bits into the dut->x bus.  Verilator exposes
        // wide buses (>32 bit) as an array of uint32_t words; for 784
        // bits that's 25 words.
        constexpr int N_WORDS = (IN_DIM + 31) / 32;
        for (int w = 0; w < N_WORDS; ++w) dut->x[w] = 0;
        for (int b = 0; b < IN_DIM; ++b) {
            if (bits[b]) {
                dut->x[b >> 5] |= (1u << (b & 31));
            }
        }
        dut->eval();

        // Read the one-hot prediction port.  We trust the comparator
        // tree to set exactly one bit, but to be safe we scan from the
        // LSB and report the first set index (matches PyTorch argmax
        // tie-break semantics by construction).
        uint16_t pred_oh = (uint16_t)dut->prediction;
        int pred = -1;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            if ((pred_oh >> c) & 1) { pred = c; break; }
        }
        if (pred < 0) {
            fprintf(stderr, "harness: no bit set in prediction at sample %d "
                            "(prediction=0x%04x)\n", s, pred_oh);
            pred = 0;
        }
        preds[s] = (uint8_t)pred;
    }

    if (fwrite(preds.data(), 1, N_SAMPLES, stdout) != (size_t)N_SAMPLES) {
        fprintf(stderr, "harness: short write\n");
        delete dut;
        return 3;
    }
    delete dut;
    return 0;
}
"""


def pytorch_bfs_predict_batch(client_models, selected, x_binary_batch, num_classes=10):
    """Run BFS prediction in PyTorch on a batch of binarized inputs (matches verify.py)."""
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
    gates = torch.cat(all_gates, dim=1)
    logits = torch.zeros(x_binary_batch.shape[0], num_classes)
    for c in range(num_classes):
        logits[:, c] = gates[:, selected[c]].sum(dim=1)
    return logits.argmax(dim=1).cpu().numpy(), logits.cpu().numpy()


def train_clients(K=5, iters=500, alpha=0.5, seed=42):
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    device = torch.device('cpu')
    train_ds, test_ds = get_mnist()
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False)
    probe_loader = torch.utils.data.DataLoader(train_ds, batch_size=1000, shuffle=False)
    client_idx = partition_dirichlet(train_ds, K, alpha=alpha)
    global_model = make_difflogic_net(784, 500, 3, 10, tau=10.0)
    client_models = [copy.deepcopy(global_model) for _ in range(K)]
    print(f"Training {K} clients ({iters} iters each)...")
    for i in range(K):
        subset = torch.utils.data.Subset(train_ds, client_idx[i])
        loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True, drop_last=True)
        train_one_client(client_models[i], loader, iters, lr=0.01, device=device,
                         binarize=True, flat=True)
    return client_models, train_ds, test_ds, probe_loader, test_loader, device


def verilate_and_run(verilog_path, in_dim, n_samples, input_bytes):
    """Build a Verilator simulation of the BFS circuit and run it.

    Returns the per-sample predicted class indices as an np.int64 array
    of length n_samples.
    """
    workdir = tempfile.mkdtemp(prefix='bfs_verilator_')
    try:
        # Copy the verilog file in so Verilator's relative paths stay tidy.
        v_local = os.path.join(workdir, 'bfs_circuit.v')
        shutil.copyfile(verilog_path, v_local)

        cpp_local = os.path.join(workdir, 'sim_main.cpp')
        with open(cpp_local, 'w') as f:
            f.write(HARNESS_CPP
                    .replace('__IN_DIM__', str(in_dim))
                    .replace('__N_SAMPLES__', str(n_samples)))

        print(f"  Verilator workdir: {workdir}")
        t0 = time.time()
        # `--cc` for C++ output, `--exe` to drive its own main, `-O3` for
        # the generated sim, `-Wno-fatal` so any innocent style warnings
        # don't kill the build.
        res = subprocess.run(
            ['verilator', '--cc', '--exe', '--build',
             '-Wno-fatal', '-Wno-WIDTHEXPAND', '-Wno-UNUSEDSIGNAL',
             '-O3',
             '--top-module', 'bfs_circuit',
             '-Mdir', os.path.join(workdir, 'obj_dir'),
             v_local, cpp_local],
            capture_output=True, text=True, timeout=600, cwd=workdir)
        if res.returncode != 0:
            sys.stderr.write(res.stdout)
            sys.stderr.write(res.stderr)
            raise RuntimeError(f"verilator build failed (returncode {res.returncode})")
        print(f"  verilator build: {time.time()-t0:.1f}s")

        sim_bin = os.path.join(workdir, 'obj_dir', 'Vbfs_circuit')
        if not os.path.exists(sim_bin):
            raise RuntimeError(f"verilator binary not found at {sim_bin}")

        t0 = time.time()
        proc = subprocess.run([sim_bin], input=input_bytes,
                              capture_output=True, timeout=600)
        print(f"  verilator run on {n_samples} samples: {time.time()-t0:.2f}s")
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr.decode('utf-8', errors='replace'))
            raise RuntimeError(f"verilator sim failed (returncode {proc.returncode})")
        if len(proc.stdout) != n_samples:
            raise RuntimeError(
                f"verilator sim returned {len(proc.stdout)} bytes, expected {n_samples}")
        return np.frombuffer(proc.stdout, dtype=np.uint8).astype(np.int64).copy()
    finally:
        # Keep the workdir on failure for easier debugging; on success
        # drop it so we don't leak temp files.
        if 'KEEP_VERILATOR_WORKDIR' not in os.environ:
            shutil.rmtree(workdir, ignore_errors=True)


def main():
    K = 5
    N = 200  # match verify.py
    NUM_SELECT = 100

    client_models, train_ds, test_ds, probe_loader, test_loader, device = \
        train_clients(K=K, iters=500)

    print("Running BFS...")
    acc, selected = boolean_feature_selection(
        client_models, probe_loader, test_loader, device,
        num_select=NUM_SELECT, binarize=True, flat=True)
    print(f"  PyTorch BFS reference accuracy: {100*acc:.2f}%")

    # Export Verilog with the now-wired argmax port.
    workdir = tempfile.mkdtemp(prefix='bfs_export_')
    v_path = os.path.join(workdir, 'bfs_circuit.v')
    print(f"\nExporting Verilog to {v_path}...")
    export_bfs_circuit(client_models, selected, num_classes=10, tau=10.0,
                       output_path=v_path)

    # Build PyTorch reference predictions for the first N test images.
    xs = torch.stack([test_ds[i][0].view(-1) for i in range(N)], dim=0)
    xs_bin = (xs > 0.5).float()
    pt_preds, pt_logits = pytorch_bfs_predict_batch(client_models, selected, xs_bin)

    # Pack inputs as raw bytes (one byte per bit, value 0 or 1) for the
    # harness to slurp from stdin.
    in_dim = xs_bin.shape[1]
    input_bytes = xs_bin.numpy().astype(np.uint8).tobytes()

    print(f"\nVerilating bfs_circuit.v and running on {N} MNIST test images...")
    v_preds = verilate_and_run(v_path, in_dim, N, input_bytes)

    matches = int((v_preds == pt_preds).sum())
    print(f"\n  {matches}/{N} predictions match between PyTorch and Verilated circuit.")
    if matches < N:
        mis = np.where(v_preds != pt_preds)[0][:5]
        for i in mis:
            print(f"    mismatch at sample {i}: pytorch={pt_preds[i]} verilator={v_preds[i]}")
            print(f"      pytorch logits: {pt_logits[i]}")

    if matches == N:
        print("\n*** PASS: Verilog round-trip correctness verified end-to-end. ***")
        return 0
    else:
        print("\n*** FAIL: Verilog predictions diverged from PyTorch reference. ***")
        return 1


if __name__ == '__main__':
    sys.exit(main())
