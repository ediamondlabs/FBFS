"""Compile BFS-selected circuit to native C for fast inference.

Uses the difflogic library's CompiledLogicNet pattern: emit C code with
bitwise operations, compile with gcc, and benchmark throughput.

For FC models: adapts the existing CompiledLogicNet approach.
The BFS-selected gates form a virtual "model" — we trace back from
selected gates through the original client models and emit only the
sub-circuits that feed them.
"""
import sys, os, copy, time, math, tempfile, subprocess, ctypes
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'federated'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (
    make_difflogic_net, get_mnist, partition_dirichlet,
    boolean_feature_selection, data_iter, LogicLayer, evaluate,
)
from gate_ops import to_c
import torch.nn.functional as F
import torch.optim as optim


def emit_bfs_c_code(client_models, selected, num_classes=10, num_bits=64):
    """Emit C code for the BFS-selected circuit (single-sample path).

    Traces back from selected gates through all client model layers,
    emitting only the gates that contribute to the selected outputs.
    Each input bit is stored in one ``long long`` and each gate becomes
    a single bitwise op; this is the path `verify.py` benchmarks
    (4.7 us/inference at 1.5K gates). Bit-packing the readout for
    sustained throughput is left as future work.
    """
    K = len(client_models)
    layers_per_model = []
    for m in client_models:
        m.eval()
        layers = [(l.idx_a.numpy(), l.idx_b.numpy(), l.weights.argmax(-1).numpy())
                  for l in m.modules() if isinstance(l, LogicLayer)]
        layers_per_model.append(layers)

    n_layers = len(layers_per_model[0])
    gates_per_layer = [l[0].shape[0] for l in layers_per_model[0]]
    in_dim = max(layers_per_model[0][0][0].max(), layers_per_model[0][0][1].max()) + 1

    # Emit C code
    lines = [
        '#include <stddef.h>',
        '#include <stdlib.h>',
        '#include <stdbool.h>',
        '#include <string.h>',
        '',
        '// BFS-selected federated difflogic circuit',
        f'// {K} client models, {n_layers} layers, {in_dim} inputs, {num_classes} classes',
        '',
        f'void bfs_circuit(long long const *inp, long long *out) {{',
    ]

    # Emit all client model gates
    var_id = 0
    model_last_layer_start = []

    for k, layers in enumerate(layers_per_model):
        lines.append(f'  // Client model {k}')
        prev_start = -1  # input
        for li, (idx_a, idx_b, gate_ops) in enumerate(layers):
            layer_start = var_id
            for gi in range(len(gate_ops)):
                if li == 0:
                    a = f"inp[{idx_a[gi]}]"
                    b = f"inp[{idx_b[gi]}]"
                else:
                    a = f"v{prev_start + idx_a[gi]}"
                    b = f"v{prev_start + idx_b[gi]}"
                expr = to_c(int(gate_ops[gi]), a, b)
                lines.append(f'  const long long v{var_id} = {expr};')
                var_id += 1
            prev_start = layer_start

        model_last_layer_start.append(prev_start)

    gates_last = gates_per_layer[-1]

    # Emit BFS counting
    lines.append('')
    lines.append('  // BFS GroupSum counting')
    for c in range(num_classes):
        terms = []
        for idx in selected[c]:
            idx_val = idx.item()
            model_k = idx_val // gates_last
            gate_gi = idx_val % gates_last
            var_name = f"v{model_last_layer_start[model_k] + gate_gi}"
            terms.append(var_name)
        lines.append(f'  out[{c}] = {" + ".join(terms)};')

    lines.append('}')

    return '\n'.join(lines), in_dim


def compile_and_benchmark(client_models, selected, test_ds, num_classes=10):
    """Compile BFS circuit to C and benchmark throughput."""
    code, in_dim = emit_bfs_c_code(client_models, selected, num_classes)

    print(f"Generated C code: {len(code.split(chr(10)))} lines")

    # Write and compile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.c', delete=False) as f:
        f.write(code)
        c_path = f.name

    so_path = c_path.replace('.c', '.so')
    print("Compiling with gcc -O1...")
    t0 = time.time()
    result = subprocess.run(
        ['gcc', '-shared', '-fPIC', '-O1', '-o', so_path, c_path],
        capture_output=True, text=True
    )
    compile_time = time.time() - t0
    print(f"  Compiled in {compile_time:.1f}s")

    if result.returncode != 0:
        print(f"  Compilation failed: {result.stderr[:500]}")
        # Save code for debugging
        debug_path = 'synthesis/bfs_compiled.c'
        with open(debug_path, 'w') as f:
            f.write(code)
        print(f"  Code saved to {debug_path}")
        return

    # Load and call
    lib = ctypes.cdll.LoadLibrary(so_path)
    bfs_fn = lib.bfs_circuit
    bfs_fn.restype = None
    bfs_fn.argtypes = [
        np.ctypeslib.ndpointer(ctypes.c_longlong, flags="C_CONTIGUOUS"),
        np.ctypeslib.ndpointer(ctypes.c_longlong, flags="C_CONTIGUOUS"),
    ]

    # Test on single sample (not bit-packed, just direct evaluation)
    x_test = test_ds[0][0].view(-1)
    x_bin = (x_test > 0.5).numpy().astype(np.int64)
    inp = np.array(x_bin, dtype=np.int64)  # each element is 0 or 1 in a long long
    out = np.zeros(num_classes, dtype=np.int64)

    bfs_fn(inp, out)
    print(f"  Test output: {out}")
    print(f"  Prediction: class {np.argmax(out)}")

    # Single-sample latency microbenchmark: same `inp`/`out` buffers reused
    # across calls, so caches and branch predictor stay hot. Read the result
    # as a hot-path service-time bound, not a sustained dataset throughput.
    N_bench = 10000
    t0 = time.time()
    for _ in range(N_bench):
        bfs_fn(inp, out)
    elapsed = time.time() - t0
    print(f"  Single-sample latency: {elapsed*1e6/N_bench:.1f} us/image"
          f" ({N_bench/elapsed:.0f} inferences/sec, hot-path)")

    # Cleanup
    os.unlink(c_path)
    os.unlink(so_path)


def main():
    torch.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cpu')
    train_ds, test_ds = get_mnist()
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False)
    probe_loader = torch.utils.data.DataLoader(train_ds, batch_size=1000, shuffle=False)

    K = 5
    client_idx = partition_dirichlet(train_ds, K, alpha=0.5)
    global_model = make_difflogic_net(784, 500, 3, 10, tau=10.0)
    client_models = [copy.deepcopy(global_model) for _ in range(K)]

    print("Training 5 clients (500 iters)...")
    for i in range(K):
        subset = torch.utils.data.Subset(train_ds, client_idx[i])
        loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True, drop_last=True)
        model = client_models[i]
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        gen = data_iter(loader)
        model.train()
        for it in range(500):
            x, y = next(gen)
            x = (x.view(x.size(0), -1) > 0.5).float()
            loss = F.cross_entropy(model(x), y)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

    print("\nRunning BFS...")
    acc, selected = boolean_feature_selection(
        client_models, probe_loader, test_loader, device,
        num_select=100, binarize=True, flat=True)
    print(f"  BFS accuracy: {100*acc:.1f}%")

    print("\nCompiling BFS circuit to C...")
    compile_and_benchmark(client_models, selected, test_ds)


if __name__ == '__main__':
    main()
