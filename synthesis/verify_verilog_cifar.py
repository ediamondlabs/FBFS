"""End-to-end Verilog verification of the BFS-selected CIFAR conv-small circuit.

Mirror of synthesis/verify_verilog.py but for the CIFAR conv-small path:

    1. Load the K=2 conv-small CIFAR clients from
       checkpoints/runs/b1_cifar10_iid_K2/.
    2. Replicate exactly the BFS selection that export_cifar.py used so the
       PyTorch reference matches the synthesized Verilog netlist.
    3. Verilate (`verilator --cc --exe --build -O3`) the regenerated
       synthesis/bfs_circuit_cifar_small.v plus a tiny C++ harness that
       reads 9216 binarized input bits per image from stdin, drives `x`,
       and writes the argmax class index to stdout.
    4. Drive N CIFAR-10 test images through the harness and compare the
       Verilator predictions to the PyTorch BFS reference. Print
       `*** PASS ***` if all match.

Single source of truth for gate semantics still lives in
`synthesis/gate_ops.py`; this script just confirms that the conv export
path is also wired correctly end-to-end.
"""
import sys, os, copy, random, tempfile, subprocess, time, shutil
import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_FED = os.path.join(_HERE, '..', 'federated')
# Order matters: there's a stale copy of export_verilog.py under federated/
# that will shadow the real one in synthesis/ if federated/ comes first.
sys.path.insert(0, _FED)
sys.path.insert(0, _HERE)

from common import (
    get_cifar10, make_conv_difflogic_net, boolean_feature_selection,
    LogicLayer,
)


HARNESS_CPP = r"""
// Verilator harness for the BFS-selected CIFAR conv-small circuit.
// Reads (IN_DIM*N_SAMPLES) bytes of {0,1} input bits from stdin, drives
// the simulated `bfs_circuit` once per sample, and writes one byte per
// sample (argmax class index) to stdout.
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
        // Pack the IN_DIM bits into the dut->x bus. Verilator exposes
        // wide buses (>32 bit) as an array of uint32_t words.
        constexpr int N_WORDS = (IN_DIM + 31) / 32;
        for (int w = 0; w < N_WORDS; ++w) dut->x[w] = 0;
        for (int b = 0; b < IN_DIM; ++b) {
            if (bits[b]) {
                dut->x[b >> 5] |= (1u << (b & 31));
            }
        }
        dut->eval();

        // Read the one-hot prediction port. The comparator tree is
        // designed to set exactly one bit, matching torch.argmax tie
        // semantics. Scan from the LSB to be safe.
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


def pytorch_bfs_predict_batch(client_models, selected, x_batch_4d, num_classes=10):
    """Run BFS prediction in PyTorch on a batch of binarized 4D inputs.

    x_batch_4d: (B, 9, 32, 32) binarized {0,1} float tensor.
    Returns (preds, logits) as numpy arrays.
    """
    all_gates = []
    for m in client_models:
        m.eval()
        layers = [mod for mod in m.modules() if isinstance(mod, LogicLayer)]
        last = layers[-1]
        hook_out = []
        h = last.register_forward_hook(lambda mm, inp, out: hook_out.append(out.detach()))
        with torch.no_grad():
            m(x_batch_4d)
        h.remove()
        all_gates.append(hook_out[0])
    gates = torch.cat(all_gates, dim=1)
    logits = torch.zeros(x_batch_4d.shape[0], num_classes)
    for c in range(num_classes):
        logits[:, c] = gates[:, selected[c]].sum(dim=1)
    return logits.argmax(dim=1).cpu().numpy(), logits.cpu().numpy()


def load_clients_and_run_bfs(seed=42):
    """Replicate exactly the setup in synthesis/export_cifar.py so the
    `selected` we get back matches what bfs_circuit_cifar_small.v was
    synthesised from (same models, same probe subsample, same BFS path).
    """
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    device = torch.device('cpu')
    K = 2
    num_select = 100

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    ckpt_dir = os.path.join(repo_root, 'scripts', 'checkpoints', 'b1_cifar10_iid_K2')

    print(f"Building conv-small models (K={K}) and loading checkpoints from")
    print(f"  {ckpt_dir}")
    client_models = []
    for k in range(K):
        m = make_conv_difflogic_net(classes=10, tau=10.0, depth=2, size='small').to(device)
        sd = torch.load(os.path.join(ckpt_dir, f'client_{k}.pt'), map_location=device)
        m.load_state_dict(sd)
        m.eval()
        client_models.append(m)
        print(f"  Loaded client {k}")

    print("\nLoading CIFAR-10 (unflattened, 9-channel binarized)...")
    train_ds, test_ds = get_cifar10(flatten=False)

    probe_n = 4000
    probe_idx = torch.randperm(len(train_ds))[:probe_n].tolist()
    probe_ds = torch.utils.data.Subset(train_ds, probe_idx)
    probe_loader = torch.utils.data.DataLoader(probe_ds, batch_size=256, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=256, shuffle=False)

    print(f"\nRunning BFS (num_select={num_select}, probe_n={probe_n})...")
    acc, selected = boolean_feature_selection(
        client_models, probe_loader, test_loader, device,
        num_select=num_select, binarize=False, flat=False)
    print(f"  PyTorch BFS reference accuracy: {100*acc:.2f}%")

    return client_models, test_ds, selected, acc


def verilate_and_run(verilog_path, in_dim, n_samples, input_bytes,
                     build_timeout=900, sim_timeout=600):
    """Build a Verilator simulation of the BFS circuit and run it.

    Returns the per-sample predicted class indices as an np.int64 array.
    """
    workdir = tempfile.mkdtemp(prefix='bfs_verilator_cifar_')
    try:
        v_local = os.path.join(workdir, 'bfs_circuit.v')
        shutil.copyfile(verilog_path, v_local)

        cpp_local = os.path.join(workdir, 'sim_main.cpp')
        with open(cpp_local, 'w') as f:
            f.write(HARNESS_CPP
                    .replace('__IN_DIM__', str(in_dim))
                    .replace('__N_SAMPLES__', str(n_samples)))

        print(f"  Verilator workdir: {workdir}")
        t0 = time.time()
        # The CIFAR conv circuit is large (~870K-line Verilog file with
        # ~435K assigns; Yosys synthesises it to 28K cells). We turn off
        # some style warnings and bump Verilator's internal limits so it
        # doesn't choke on the giant top module.
        res = subprocess.run(
            ['verilator', '--cc', '--exe', '--build',
             '-Wno-fatal', '-Wno-WIDTHEXPAND', '-Wno-UNUSEDSIGNAL',
             '-Wno-UNUSEDPARAM', '-Wno-VARHIDDEN',
             '-O3',
             '--top-module', 'bfs_circuit',
             '-Mdir', os.path.join(workdir, 'obj_dir'),
             v_local, cpp_local],
            capture_output=True, text=True, timeout=build_timeout, cwd=workdir)
        if res.returncode != 0:
            sys.stderr.write(res.stdout)
            sys.stderr.write(res.stderr)
            raise RuntimeError(f"verilator build failed (returncode {res.returncode})")
        build_time = time.time() - t0
        print(f"  verilator build: {build_time:.1f}s")

        sim_bin = os.path.join(workdir, 'obj_dir', 'Vbfs_circuit')
        if not os.path.exists(sim_bin):
            raise RuntimeError(f"verilator binary not found at {sim_bin}")

        t0 = time.time()
        proc = subprocess.run([sim_bin], input=input_bytes,
                              capture_output=True, timeout=sim_timeout)
        sim_time = time.time() - t0
        print(f"  verilator run on {n_samples} samples: {sim_time:.2f}s")
        if proc.returncode != 0:
            sys.stderr.write(proc.stderr.decode('utf-8', errors='replace'))
            raise RuntimeError(f"verilator sim failed (returncode {proc.returncode})")
        if len(proc.stdout) != n_samples:
            raise RuntimeError(
                f"verilator sim returned {len(proc.stdout)} bytes, expected {n_samples}")
        return (np.frombuffer(proc.stdout, dtype=np.uint8).astype(np.int64).copy(),
                build_time, sim_time)
    finally:
        if 'KEEP_VERILATOR_WORKDIR' not in os.environ:
            shutil.rmtree(workdir, ignore_errors=True)


def main():
    N = int(os.environ.get('VERIFY_N', '200'))

    client_models, test_ds, selected, bfs_acc = load_clients_and_run_bfs(seed=42)

    v_path = os.path.join(os.path.dirname(__file__), 'bfs_circuit_cifar_small.v')
    if not os.path.exists(v_path):
        raise RuntimeError(f"Expected synthesized Verilog at {v_path}; "
                           f"run synthesis/export_cifar.py first.")

    # Build PyTorch reference predictions for the first N test images.
    # CIFAR conv expects unflattened (9, 32, 32) per sample, already
    # binarized via the get_cifar10 transform.
    xs = torch.stack([test_ds[i][0] for i in range(N)], dim=0)  # (N, 9, 32, 32)
    assert xs.shape[1:] == (9, 32, 32), f"unexpected shape {xs.shape}"
    pt_preds, pt_logits = pytorch_bfs_predict_batch(client_models, selected, xs)

    # Pack input bytes for the harness. The exporter places channel ch,
    # row h, col w at flat index ch*32*32 + h*32 + w, which is exactly
    # what x.view(-1) produces on a (9, 32, 32) tensor.
    in_dim = 9 * 32 * 32  # 9216
    flat = xs.view(N, in_dim).numpy().astype(np.uint8)
    # Sanity: input is binary
    assert flat.max() <= 1 and flat.min() >= 0
    input_bytes = flat.tobytes()

    print(f"\nVerilating bfs_circuit_cifar_small.v and running on {N} CIFAR-10 test images...")
    v_preds, build_time, sim_time = verilate_and_run(v_path, in_dim, N, input_bytes)

    matches = int((v_preds == pt_preds).sum())
    print(f"\n  {matches}/{N} predictions match between PyTorch and Verilated circuit.")
    print(f"  PyTorch BFS reference accuracy on full test set: {100*bfs_acc:.2f}%")
    print(f"  Verilator build time:    {build_time:.1f}s")
    print(f"  Verilator sim time:      {sim_time:.2f}s ({n_per_sec(sim_time, N)})")

    if matches < N:
        mis = np.where(v_preds != pt_preds)[0][:5]
        for i in mis:
            print(f"    mismatch at sample {i}: pytorch={pt_preds[i]} verilator={v_preds[i]}")
            print(f"      pytorch logits: {pt_logits[i]}")

    if matches == N:
        print("\n*** PASS: CIFAR Verilog round-trip correctness verified end-to-end. ***")
        return 0
    else:
        print("\n*** FAIL: CIFAR Verilog predictions diverged from PyTorch reference. ***")
        return 1


def n_per_sec(t, n):
    if t <= 0:
        return "inf img/s"
    return f"{n/t:.1f} img/s"


if __name__ == '__main__':
    sys.exit(main())
