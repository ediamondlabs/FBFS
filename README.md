# FBFS

Implementation of **Federated Logic Gate Networks via Boolean Feature
Selection**, by Felix Kunze and Antonio Di Maio (NeurIPS 2026), and the
accompanying thesis. Repository: <https://github.com/ediamondlabs/FBFS>.

FBFS selects last-layer Boolean outputs from client networks and retains
their upstream circuits. Clients share firing statistics, not raw examples.
The implementation includes basic and diversity-aware selection, baseline
merges, and C/Verilog export.

## Setup

Python 3.12; run from this directory:

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -s tests
python smoke_artifact.py
```

The smoke test uses synthetic data, checks pooled/federated selection, and
verifies compiled C against PyTorch. A C compiler is required; `--skip-c`
omits compilation. The pinned environment is the CPU/MPS reference. For CUDA,
install a PyTorch/torchvision pair compatible with the host GPU and driver,
then `pip install -r requirements-gpu.txt`, which does not repin PyTorch.
LogicTreeNet-S/B need CUDA; full-pool B selection
used a 96 GB GPU. Verilog verification needs Verilator; synthesis needs Yosys.

## Train and Evaluate

```sh
# Shared-wiring baseline comparison
python federated/local_lgn.py --split dirichlet --alpha 0.5 --no-plot

# Held-out FBFS and diverse FBFS
python scripts/multiseed_alpha_bfs.py --alphas 0.1 --seeds 42 \
  --out scripts/results/fbfs.json

# Deploy and verify a trained MNIST circuit
python synthesis/verify.py
python synthesis/verify_verilog.py
```

`federated/common.py` contains the model layers, training helpers, merge
rules, and FBFS protocols. `synthesis/` contains circuit export and verification.

Only the dataset-specific experiment entry points are retained:

| Benchmark | Driver |
| --- | --- |
| MNIST | `scripts/multiseed_alpha_bfs.py` |
| FEMNIST | `scripts/femnist.py` |
| UCI HAR / HHAR | `scripts/har.py`, `scripts/hhar.py` |
| ExtraSensory | `scripts/extrasensory.py` |
| Fed-ISIC2019 | `scripts/fedisic.py` |
| CIFAR-10, LogicTreeNet-S / B | `scripts/logictreenet_s.py`, `scripts/logictreenet_b.py` |

Each exposes its configuration through `--help`. `full_pool_bfs.py` is the
shared B-selection helper. Configurations and seed counts are in the paper.
MNIST, CIFAR-10, FEMNIST, and HAR download through their drivers. HHAR,
ExtraSensory, and Fed-ISIC2019 require the providers' files at the paths
specified by their drivers. ExtraSensory's paper setting uses
`--feature-set all --neurons 600`. S/B diagnostics use centralized training
probes, not the held-out federated protocol.

This compact release omits historical sweeps, rebuttal-only diagnostics,
cloud orchestration, and manuscript plotting. Those remain in the development
workspace; this is not an automated reproduction of every appendix table.
Datasets, checkpoints, and archived results are not bundled. The fixed-circuit
first-moment privacy helper is not end-to-end DP. See `CITATION.cff` and `LICENSE`.
