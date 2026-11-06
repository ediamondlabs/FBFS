"""Check federated selection and compiled C on synthetic data; no downloads."""

import argparse
import ctypes
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset


ROOT = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT / "federated"), str(ROOT / "synthesis")]
from common import make_difflogic_net, boolean_feature_selection, bfs_federated
from compiled_bfs import emit_bfs_c_code
from verify import pytorch_bfs_predict_batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-c", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(42)
    models = [make_difflogic_net(8, 12, 2, 2).eval() for _ in range(2)]
    x = torch.randint(0, 2, (128, 8)).float()
    dataset = TensorDataset(x, torch.arange(128) % 2)
    loader = DataLoader(dataset, batch_size=32)
    clients = [DataLoader(Subset(dataset, range(a, b)), batch_size=32)
               for a, b in [(0, 64), (64, 128)]]
    central, selected = boolean_feature_selection(
        models, loader, loader, "cpu", num_classes=2, num_select=3, binarize=False)
    federated, other = bfs_federated(
        models, clients, loader, "cpu", num_classes=2, num_select=3, binarize=False)
    assert central == federated
    assert all(torch.equal(selected[c], other[c]) for c in selected)
    print("PASS: pooled and federated selection agree.")
    if args.skip_c:
        return
    compiler = shutil.which("gcc") or shutil.which("cc")
    if not compiler:
        parser.error("Install a C compiler or use --skip-c.")
    code, _ = emit_bfs_c_code(models, selected, num_classes=2)
    _, expected = pytorch_bfs_predict_batch(models, selected, x, num_classes=2)
    with tempfile.TemporaryDirectory(prefix="fbfs-smoke-") as directory:
        source, library = Path(directory) / "circuit.c", Path(directory) / "circuit.so"
        source.write_text(code)
        subprocess.run([compiler, "-shared", "-fPIC", "-O1", "-o", str(library), str(source)], check=True)
        lib = ctypes.cdll.LoadLibrary(str(library))
        function = lib.bfs_circuit
        function.restype = None
        function.argtypes = [np.ctypeslib.ndpointer(ctypes.c_longlong, flags="C_CONTIGUOUS")] * 2
        for i, row in enumerate(x.numpy().astype(np.int64)):
            output = np.zeros(2, dtype=np.int64)
            function(row, output)
            assert np.array_equal(output, expected[i]), (i, output, expected[i])
    print("PASS: compiled C matches PyTorch class counts on 128 inputs.")


if __name__ == "__main__":
    main()
