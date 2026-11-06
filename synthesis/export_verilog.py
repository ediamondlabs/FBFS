"""Export BFS-selected difflogic circuit to Verilog.

Takes K trained client models + BFS selection indices, traces backward
from selected gates, and emits a single Verilog netlist. Optionally
runs Yosys for dead logic elimination and reports gate count.

Supports:
  - native FC (LogicLayer)
  - native conv (LogicTreeConv2d)
  - native pooling (ORPool2d)
  - torchlogix FC (LogicDense), conv (LogicConv2d), pool (TLOrPool)
"""
import sys, os, copy, argparse
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'federated'))
from common import LogicLayer, LogicTreeConv2d, ORPool2d, GroupSum
from gate_ops import to_verilog, GATE_NAMES

# torchlogix shims (paper_s arch). Imported lazily so this script still works
# in environments without torchlogix installed.
try:
    from torchlogix.layers.conv import LogicConv2d as _TLLogicConv2d
    from torchlogix.layers.dense import LogicDense as _TLLogicDense
    from torchlogix.layers.groupsum import GroupSum as _TLGroupSum
    from torchlogix.layers.pool import OrPooling2d as _TLOrPooling2d
    _HAS_TL = True
except ImportError:
    _TLLogicConv2d = _TLLogicDense = _TLGroupSum = _TLOrPooling2d = None
    _HAS_TL = False


def _tl_conv_emit(mod, model_id, lines, signals, spatial_shape, layer_count):
    """Emit Verilog for a torchlogix LogicConv2d.

    Returns updated (signals, spatial_shape, total_gates_added).

    Tree weights live in mod.tree_weights, a ParameterList of length tree_depth.
    Per torchlogix init, weights[0] is the level-0 (leaf-pair) gates,
    indexed (lut_rank**(tree_depth-1), num_kernels, 16) -> argmax(-1) gives
    the gate type as a 0..15 ID using the same op ordering as our gate_ops.

    Connections live in mod.connections.indices, list of length tree_depth.
    indices[0] has shape (lut_rank, num_kernels, num_positions, S, 3) with
    last dim = (h_padded, w_padded, c) of the leaf inputs (h/w in padded
    coords, so subtract mod.padding to get input-image coords; out-of-bounds
    means zero padding).  indices[level>=1] is just (lut_rank, output_size)
    encoding intra-tree pair-wise wiring.
    """
    if not _HAS_TL:
        raise RuntimeError("torchlogix not installed; cannot export paper_s arch")
    if spatial_shape is None:
        # First conv layer: derive (C, H, W) from the layer's own in_dim/channels.
        H, W = mod.in_dim
        C_in = mod.channels
    else:
        C_in, H, W = spatial_shape
    K_kernels = mod.num_kernels
    td = mod.tree_depth
    lut_rank = mod.lut_rank
    pad = mod.padding
    rfs = mod.receptive_field_size  # tuple
    # We support stride=1 only (matches all paper_s/paper_b configs).
    stride = mod.stride if isinstance(mod.stride, int) else mod.stride[0]
    assert stride == 1, f"torchlogix conv export only supports stride=1, got {stride}"
    H_out = (H + 2 * pad - rfs[0]) // stride + 1
    W_out = (W + 2 * pad - rfs[1]) // stride + 1

    gate_ids_per_level = []
    for level in range(td):
        w = mod.tree_weights[level]  # (n_gates_in_level, num_kernels, 16)
        gate_ids_per_level.append(w.argmax(-1))  # (n_gates_in_level, num_kernels)

    indices = mod.connections.indices  # list, indices[0] is (L, K, P, S, 3)
    idx0 = indices[0]
    # P = num_positions; should equal H_out * W_out
    L_, K_, P, S, _ = idx0.shape
    assert L_ == lut_rank and K_ == K_kernels and P == H_out * W_out, \
        f"indices0 shape {idx0.shape} doesn't match expected ({lut_rank},{K_kernels},{H_out*W_out},?,3)"

    # Helper: input wire from current `signals` dict (indexed by (c, h, w)
    # for intermediate conv outputs).  On the FIRST conv (signals empty),
    # the fallback subscripts the flat input bus m{k}_input[idx] using the
    # CIFAR convention flat = c * H * W + h * W + w (matches input wiring
    # set up by export_bfs_circuit).
    is_first_conv = not any(isinstance(k, tuple) for k in signals.keys())
    def in_wire(c, src_h, src_w):
        if 0 <= src_h < H and 0 <= src_w < W:
            if is_first_conv:
                flat = c * H * W + src_h * W + src_w
                return signals.get((c, src_h, src_w),
                                   f"m{model_id}_input[{flat}]")
            return signals.get((c, src_h, src_w),
                               f"m{model_id}_inp_{c}_{src_h}_{src_w}")
        return "1'b0"

    new_signals = {}
    total_gates_added = 0

    for k in range(K_kernels):
        for p in range(P):
            hi = p // W_out
            wi = p % W_out
            # === Level 0 leaf-pair gates (S of them) ===
            level_outputs = []  # list of S wire names from this level
            for s in range(S):
                # L=0 -> child a, L=1 -> child b
                # idx0[L, k, p, s] = (h_padded, w_padded, c)
                ah_p, aw_p, ac = (idx0[0, k, p, s, 0].item(),
                                  idx0[0, k, p, s, 1].item(),
                                  idx0[0, k, p, s, 2].item())
                bh_p, bw_p, bc = (idx0[1, k, p, s, 0].item(),
                                  idx0[1, k, p, s, 1].item(),
                                  idx0[1, k, p, s, 2].item())
                a_wire = in_wire(ac, ah_p - pad, aw_p - pad)
                b_wire = in_wire(bc, bh_p - pad, bw_p - pad)
                gt = gate_ids_per_level[0][s, k].item()
                wire_name = f"m{model_id}_tlconv{layer_count}_k{k}_p{p}_l0_s{s}"
                expr = to_verilog(gt, a_wire, b_wire)
                lines.append(f"  wire {wire_name};")
                lines.append(f"  assign {wire_name} = {expr}; // {GATE_NAMES[gt]}")
                level_outputs.append(wire_name)
                total_gates_added += 1
            # === Levels 1..td-1: intra-tree pairing per torchlogix indices ===
            for level in range(1, td):
                size = lut_rank ** (td - level)  # number of *inputs* at this level
                idx_l = indices[level]  # (lut_rank, output_size)
                output_size = size // lut_rank
                next_level_outputs = [None] * output_size
                for og in range(output_size):
                    a_wire = level_outputs[idx_l[0, og].item()]
                    b_wire = level_outputs[idx_l[1, og].item()]
                    gt = gate_ids_per_level[level][og, k].item()
                    wire_name = f"m{model_id}_tlconv{layer_count}_k{k}_p{p}_l{level}_g{og}"
                    expr = to_verilog(gt, a_wire, b_wire)
                    lines.append(f"  wire {wire_name};")
                    lines.append(f"  assign {wire_name} = {expr}; // {GATE_NAMES[gt]}")
                    next_level_outputs[og] = wire_name
                    total_gates_added += 1
                level_outputs = next_level_outputs
            assert len(level_outputs) == 1, f"tree didn't reduce to 1, got {len(level_outputs)}"
            new_signals[(k, hi, wi)] = level_outputs[0]
    return new_signals, (K_kernels, H_out, W_out), total_gates_added


def _tl_dense_emit(mod, model_id, lines, signals, layer_count):
    """Emit Verilog for a torchlogix LogicDense layer.

    weight is (out_dim, 16); argmax(-1) gives gate type.
    connections.indices is (lut_rank, out_dim), gives the lut_rank fan-in
    indices into the input feature dimension.
    """
    if not _HAS_TL:
        raise RuntimeError("torchlogix not installed")
    out_dim = mod.out_dim
    gate_ids = mod.weight.argmax(-1)  # (out_dim,)
    idx = mod.connections.indices  # (lut_rank, out_dim)
    new_signals = {}
    for gi in range(out_dim):
        a_idx = idx[0, gi].item()
        b_idx = idx[1, gi].item()
        a_wire = signals.get(a_idx, f"m{model_id}_input[{a_idx}]")
        b_wire = signals.get(b_idx, f"m{model_id}_input[{b_idx}]")
        gt = gate_ids[gi].item()
        wire_name = f"m{model_id}_tlfc{layer_count}_g{gi}"
        expr = to_verilog(gt, a_wire, b_wire)
        lines.append(f"  wire {wire_name};")
        lines.append(f"  assign {wire_name} = {expr}; // {GATE_NAMES[gt]}")
        new_signals[gi] = wire_name
    return new_signals, out_dim


def export_model(model, model_id, lines):
    """Export a difflogic model (FC or conv) to Verilog wire assignments.

    Handles LogicLayer (FC), LogicTreeConv2d (conv), ORPool2d, Flatten.
    Returns dict mapping gate_index -> wire_name for last LogicLayer gates.
    """
    # Collect all layers in order
    modules = list(model.children()) if hasattr(model, 'children') else [model]

    # Track the current "signal space" as a flat dict: index -> wire_name
    signals = {}  # current signal index -> wire name
    spatial_shape = None  # (C, H, W) for conv layers, None for FC
    layer_count = 0
    total_gates = 0

    for mod in modules:
        if isinstance(mod, nn.Flatten):
            # Flatten spatial signals to 1D
            if spatial_shape:
                C, H, W = spatial_shape
                flat_signals = {}
                for c in range(C):
                    for h in range(H):
                        for w in range(W):
                            flat_idx = c * H * W + h * W + w
                            flat_signals[flat_idx] = signals.get((c, h, w),
                                f"m{model_id}_flat_{flat_idx}")
                signals = flat_signals
                spatial_shape = None

        elif isinstance(mod, LogicLayer):
            out_dim = mod.weights.shape[0]
            gate_types = mod.weights.argmax(-1)
            idx_a = mod.idx_a
            idx_b = mod.idx_b
            new_signals = {}

            for gi in range(out_dim):
                gt = gate_types[gi].item()
                a_idx = idx_a[gi].item()
                b_idx = idx_b[gi].item()
                a_wire = signals.get(a_idx, f"m{model_id}_input[{a_idx}]")
                b_wire = signals.get(b_idx, f"m{model_id}_input[{b_idx}]")

                wire_name = f"m{model_id}_fc{layer_count}_g{gi}"
                expr = to_verilog(gt, a_wire, b_wire)
                lines.append(f"  wire {wire_name};")
                lines.append(f"  assign {wire_name} = {expr}; // {GATE_NAMES[gt]}")
                new_signals[gi] = wire_name
                total_gates += 1

            signals = new_signals
            layer_count += 1

        elif isinstance(mod, LogicTreeConv2d):
            if spatial_shape is None:
                # First conv layer — set up input signals
                # Assume input is already in signals as flat, or we create from input port
                pass  # signals already set from input or previous layer

            C_in = mod.in_channels
            C_out = mod.out_channels
            k = mod.kernel_size
            pad = mod.padding
            depth = mod.depth

            if spatial_shape:
                _, H, W = spatial_shape
            else:
                # Infer from signal count if possible
                # For first layer, signals might be input
                H = W = 32  # default for CIFAR-10

            H_out = H + 2 * pad - k + 1
            W_out = W + 2 * pad - k + 1

            gate_types = mod.weights.argmax(-1)  # (C_out, n_gates)
            conn_c = mod.conn_c  # (C_out, n_leaves)
            conn_hw = mod.conn_hw  # (C_out, n_leaves)

            new_signals = {}

            for oc in range(C_out):
                for hi in range(H_out):
                    for wi in range(W_out):
                        # Get leaf values for this spatial position
                        leaf_wires = []
                        for leaf in range(mod.n_leaves):
                            ch = conn_c[oc, leaf].item()
                            hw_off = conn_hw[oc, leaf].item()
                            dh = hw_off // k
                            dw = hw_off % k
                            src_h = hi + dh - pad
                            src_w = wi + dw - pad

                            if 0 <= src_h < H and 0 <= src_w < W:
                                if spatial_shape:
                                    leaf_wire = signals.get((ch, src_h, src_w),
                                        f"m{model_id}_input_{ch}_{src_h}_{src_w}")
                                else:
                                    flat_idx = ch * H * W + src_h * W + src_w
                                    leaf_wire = signals.get(flat_idx,
                                        f"m{model_id}_input[{flat_idx}]")
                            else:
                                leaf_wire = "1'b0"  # zero padding
                            leaf_wires.append(leaf_wire)

                        # Build tree bottom-up
                        current = leaf_wires
                        gate_idx = 0
                        for level in range(depth):
                            n_pairs = len(current) // 2
                            next_level = []
                            for p in range(n_pairs):
                                gt = gate_types[oc, gate_idx + p].item()
                                a_wire = current[2 * p]
                                b_wire = current[2 * p + 1]
                                wire_name = f"m{model_id}_conv{layer_count}_oc{oc}_h{hi}_w{wi}_d{level}_p{p}"
                                expr = to_verilog(gt, a_wire, b_wire)
                                lines.append(f"  wire {wire_name};")
                                lines.append(f"  assign {wire_name} = {expr}; // {GATE_NAMES[gt]}")
                                next_level.append(wire_name)
                                total_gates += 1
                            current = next_level
                            gate_idx += n_pairs

                        # Tree output
                        new_signals[(oc, hi, wi)] = current[0]

            signals = new_signals
            spatial_shape = (C_out, H_out, W_out)
            layer_count += 1

        elif isinstance(mod, ORPool2d):
            if spatial_shape is None:
                continue
            C, H, W = spatial_shape
            ks = mod.pool.kernel_size
            stride = mod.pool.stride
            if isinstance(ks, int):
                ks = (ks, ks)
            if isinstance(stride, int):
                stride = (stride, stride)
            H_out = H // stride[0]
            W_out = W // stride[1]

            new_signals = {}
            for c in range(C):
                for hi in range(H_out):
                    for wi in range(W_out):
                        # OR pool = max over the kernel window
                        pool_wires = []
                        for dh in range(ks[0]):
                            for dw in range(ks[1]):
                                src_h = hi * stride[0] + dh
                                src_w = wi * stride[1] + dw
                                if (c, src_h, src_w) in signals:
                                    pool_wires.append(signals[(c, src_h, src_w)])

                        if len(pool_wires) == 0:
                            wire = "1'b0"
                        elif len(pool_wires) == 1:
                            wire = pool_wires[0]
                        else:
                            # OR tree
                            wire = pool_wires[0]
                            for pi, pw in enumerate(pool_wires[1:]):
                                new_wire = f"m{model_id}_orpool{layer_count}_c{c}_h{hi}_w{wi}_p{pi}"
                                lines.append(f"  wire {new_wire};")
                                lines.append(f"  assign {new_wire} = {wire} | {pw}; // OR_POOL")
                                wire = new_wire
                                total_gates += 1

                        new_signals[(c, hi, wi)] = wire

            signals = new_signals
            spatial_shape = (C, H_out, W_out)
            layer_count += 1

        elif _HAS_TL and isinstance(mod, _TLLogicConv2d):
            new_signals, spatial_shape, n = _tl_conv_emit(
                mod, model_id, lines, signals, spatial_shape, layer_count)
            signals = new_signals
            total_gates += n
            layer_count += 1

        elif _HAS_TL and isinstance(mod, _TLOrPooling2d):
            assert spatial_shape is not None
            C, H, W = spatial_shape
            ks = mod.kernel_size
            stride = mod.stride
            if isinstance(ks, int):
                ks = (ks, ks)
            if isinstance(stride, int):
                stride = (stride, stride)
            H_out = H // stride[0]
            W_out = W // stride[1]
            new_signals = {}
            for c in range(C):
                for hi in range(H_out):
                    for wi in range(W_out):
                        pool_wires = []
                        for dh in range(ks[0]):
                            for dw in range(ks[1]):
                                src_h = hi * stride[0] + dh
                                src_w = wi * stride[1] + dw
                                if (c, src_h, src_w) in signals:
                                    pool_wires.append(signals[(c, src_h, src_w)])
                        if not pool_wires:
                            wire = "1'b0"
                        elif len(pool_wires) == 1:
                            wire = pool_wires[0]
                        else:
                            wire = pool_wires[0]
                            for pi, pw in enumerate(pool_wires[1:]):
                                new_wire = f"m{model_id}_tlorpool{layer_count}_c{c}_h{hi}_w{wi}_p{pi}"
                                lines.append(f"  wire {new_wire};")
                                lines.append(f"  assign {new_wire} = {wire} | {pw}; // OR_POOL")
                                wire = new_wire
                                total_gates += 1
                        new_signals[(c, hi, wi)] = wire
            signals = new_signals
            spatial_shape = (C, H_out, W_out)
            layer_count += 1

        elif _HAS_TL and isinstance(mod, _TLLogicDense):
            new_signals, _ = _tl_dense_emit(
                mod, model_id, lines, signals, layer_count)
            signals = new_signals
            layer_count += 1
            total_gates += mod.out_dim

        elif _HAS_TL and isinstance(mod, _TLGroupSum):
            pass  # handled externally

        elif isinstance(mod, GroupSum):
            pass  # handled externally

    # Return last-layer signals (flat FC indices)
    last_layer_map = {}
    for idx, wire in signals.items():
        if isinstance(idx, int):
            last_layer_map[idx] = wire
        # If still spatial, flatten
        elif isinstance(idx, tuple) and len(idx) == 3:
            c, h, w = idx
            C, H, W = spatial_shape
            flat_idx = c * H * W + h * W + w
            last_layer_map[flat_idx] = wire

    # Determine input size.
    candidate_classes = [LogicLayer, LogicTreeConv2d]
    if _HAS_TL:
        candidate_classes += [_TLLogicConv2d, _TLLogicDense]
    first_mod = next((m for m in model.modules()
                      if isinstance(m, tuple(candidate_classes))), None)
    if isinstance(first_mod, LogicTreeConv2d):
        in_dim = first_mod.in_channels * 32 * 32
    elif isinstance(first_mod, LogicLayer):
        in_dim = max(first_mod.idx_a.max().item(),
                     first_mod.idx_b.max().item()) + 1
    elif _HAS_TL and isinstance(first_mod, _TLLogicConv2d):
        # Initial input is unflattened (channels, H, W)
        in_dim = first_mod.channels * first_mod.in_dim[0] * first_mod.in_dim[1]
    elif _HAS_TL and isinstance(first_mod, _TLLogicDense):
        in_dim = first_mod.in_dim
    else:
        in_dim = 9 * 32 * 32  # CIFAR-10 default

    print(f"  Model {model_id}: {total_gates} gates exported")
    return last_layer_map, in_dim


def export_bfs_circuit(client_models, selected_indices, num_classes=10,
                       tau=10.0, output_path="bfs_circuit.v",
                       balanced_readout=False):
    """Export BFS-selected circuit to Verilog.

    Args:
        client_models: list of K trained models (eval mode, hardened)
        selected_indices: dict {class_c: tensor of gate indices}
            where indices are into the concatenated last-layer gate pool
        num_classes: number of output classes
        tau: GroupSum temperature
        output_path: where to write the Verilog file
        balanced_readout: emit an explicit balanced adder tree instead of one
            large expression. This is required for readouts with thousands of
            terms, which can exceed parser recursion limits in synthesis tools.
    """
    K = len(client_models)
    lines = []

    # Export each client model to determine input size
    all_last_gates = {}
    gates_per_model = None
    in_dim = None

    model_lines = []
    for k in range(K):
        ml = []
        ml.append(f"  // === Client model {k} ===")
        model = client_models[k]
        model.eval()

        last_map, model_in_dim = export_model(model, k, ml)

        if in_dim is None:
            in_dim = model_in_dim
        if gates_per_model is None:
            gates_per_model = len(last_map)

        for gi, wire in last_map.items():
            global_idx = k * gates_per_model + gi
            all_last_gates[global_idx] = wire

        ml.append(f"")
        model_lines.extend(ml)

    # Module header
    lines.append(f"// BFS-selected difflogic circuit")
    lines.append(f"// {K} client models, {num_classes} classes")
    lines.append(f"module bfs_circuit(")
    lines.append(f"  input [{in_dim-1}:0] x,")
    lines.append(f"  output [{num_classes-1}:0] prediction")
    lines.append(f");")
    lines.append(f"")

    # Declare input aliases for each model
    for k in range(K):
        lines.append(f"  wire [{in_dim-1}:0] m{k}_input;")
        lines.append(f"  assign m{k}_input = x;")
    lines.append(f"")

    # Add model lines
    lines.extend(model_lines)

    # GroupSum counters for each class using selected gates
    lines.append(f"  // === BFS GroupSum counters ===")
    for c in range(num_classes):
        sel = selected_indices[c]
        n_sel = len(sel)
        # Sum the selected gate outputs
        terms = []
        for idx in sel:
            idx_val = idx.item() if hasattr(idx, 'item') else idx
            if idx_val in all_last_gates:
                terms.append(all_last_gates[idx_val])
            else:
                terms.append(f"1'b0 /* missing gate {idx_val} */")

        # Use a simple adder tree
        bit_width = (n_sel - 1).bit_length() + 1
        lines.append(f"  wire [{bit_width-1}:0] class_{c}_count;")

        if len(terms) <= 16 or not balanced_readout:
            # Small enough for direct sum
            sum_expr = " + ".join(terms)
            lines.append(f"  assign class_{c}_count = {sum_expr};")
        else:
            # Explicit balanced tree keeps synthesis parser depth logarithmic.
            current = []
            for i, t in enumerate(terms):
                lines.append(f"  wire class_{c}_sel_{i};")
                lines.append(f"  assign class_{c}_sel_{i} = {t};")
                current.append((f"class_{c}_sel_{i}", 1))
            level = 0
            while len(current) > 1:
                next_level = []
                for pair_index in range(0, len(current), 2):
                    if pair_index + 1 == len(current):
                        next_level.append(current[pair_index])
                        continue
                    (left, left_width), (right, right_width) = current[
                        pair_index : pair_index + 2
                    ]
                    width = max(left_width, right_width) + 1
                    name = f"class_{c}_sum_l{level}_{pair_index // 2}"
                    lines.append(f"  wire [{width-1}:0] {name};")
                    lines.append(f"  assign {name} = {left} + {right};")
                    next_level.append((name, width))
                current = next_level
                level += 1
            lines.append(f"  assign class_{c}_count = {current[0][0]};")

        lines.append(f"")

    # Argmax comparator tree producing a one-hot prediction.
    # PyTorch's torch.argmax returns the LOWEST index on ties, so for
    # class c we set prediction[c] = 1 iff
    #   count[c] >  count[i] for every i < c   (strict, breaks ties down)
    #   count[c] >= count[i] for every i > c
    # Exactly one bit is set, regardless of how many counts tie.
    lines.append(f"  // === Argmax comparator tree (one-hot prediction) ===")
    lines.append(f"  // PyTorch torch.argmax returns the lowest index on ties; we match that")
    lines.append(f"  // by using strict > against lower-indexed classes and >= against higher.")
    for c in range(num_classes):
        cmp_terms = []
        for i in range(num_classes):
            if i == c:
                continue
            op = ">" if i < c else ">="
            cmp_terms.append(f"(class_{c}_count {op} class_{i}_count)")
        cmp_expr = " & ".join(cmp_terms)
        lines.append(f"  assign prediction[{c}] = {cmp_expr};")

    lines.append(f"")
    lines.append(f"endmodule")

    # Write file
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    # Stats
    total_gates = sum(1 for l in lines if 'assign' in l and 'GATE' in l.upper()
                      or any(gn in l for gn in GATE_NAMES))
    gate_assigns = sum(1 for l in lines if l.strip().startswith('assign') and '//' in l)

    print(f"Exported to {output_path}")
    print(f"  {len(lines)} lines of Verilog")
    print(f"  ~{gate_assigns} gate assignments")
    print(f"  {K} client models, {gates_per_model} gates/model")
    print(f"  {sum(len(selected_indices[c]) for c in range(num_classes))} selected outputs")

    return output_path


def run_yosys_stats(verilog_path, write_synth_to=None):
    """Run Yosys to get post-synthesis gate count.

    Returns a dict with keys 'cells', 'wires', 'public_wires', and the
    raw stdout, or None if Yosys isn't available.
    """
    import subprocess
    write_step = f"write_verilog {write_synth_to}\n" if write_synth_to else ""
    yosys_script = f"""
read_verilog {verilog_path}
hierarchy -top bfs_circuit
proc
opt -full
synth -top bfs_circuit
opt -full
{write_step}stat
"""
    try:
        # 10-minute cap: the conv circuit is an ~870K-line Verilog file
        # (~435K assign statements) and Yosys's opt passes can take several
        # minutes on it.
        result = subprocess.run(['yosys', '-p', yosys_script],
                                capture_output=True, text=True, timeout=600)
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        print(f"  Yosys not available or timed out: {e}")
        return None

    cells = wires = pub_wires = None
    in_stat = False
    # Yosys's `stat` pass prints lines like "  8798 wires", "  1838 public wires",
    # "  6970 cells".  We grab the last occurrence (post-opt) of each.
    for line in result.stdout.split('\n'):
        if 'Printing statistics.' in line:
            in_stat = True
            cells = wires = pub_wires = None  # reset for the final stat block
            continue
        if not in_stat:
            continue
        toks = line.split()
        if len(toks) < 2:
            continue
        try:
            n = int(toks[0])
        except ValueError:
            continue
        rest = ' '.join(toks[1:]).lower()
        if rest == 'wires':
            wires = n
            print(f"  Yosys: {n} wires")
        elif rest == 'public wires':
            pub_wires = n
            print(f"  Yosys: {n} public wires")
        elif rest == 'cells':
            cells = n
            print(f"  Yosys: {n} cells")
    return {'cells': cells, 'wires': wires, 'public_wires': pub_wires,
            'stdout': result.stdout}


if __name__ == '__main__':
    import copy
    from common import (get_mnist, make_difflogic_net, evaluate,
                        boolean_feature_selection, partition_dirichlet,
                        data_iter)
    import torch.nn.functional as F
    import torch.optim as optim

    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint_dir', default='./checkpoints/b1_mnist_dirichlet_K5')
    parser.add_argument('--output', default='bfs_circuit.v')
    parser.add_argument('--num_select', type=int, default=100)
    parser.add_argument('--train', action='store_true',
                        help='Train fresh models instead of loading checkpoints')
    args = parser.parse_args()

    device = torch.device('cpu')
    train_ds, test_ds = get_mnist()
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=1000, shuffle=False)
    probe_loader = torch.utils.data.DataLoader(train_ds, batch_size=1000, shuffle=False)

    K = 5
    global_model = make_difflogic_net(784, 500, 3, 10, tau=10.0).to(device)

    if args.train:
        import random, numpy as np
        torch.manual_seed(42); random.seed(42); np.random.seed(42)
        client_idx = partition_dirichlet(train_ds, K, alpha=0.5)
        client_models = [copy.deepcopy(global_model) for _ in range(K)]
        for i in range(K):
            subset = torch.utils.data.Subset(train_ds, client_idx[i])
            loader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True, drop_last=True)
            model = client_models[i]
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            gen = data_iter(loader)
            model.train()
            for it in range(500):
                x, y = next(gen)
                x, y = x.to(device), y.to(device)
                x = (x.view(x.size(0), -1) > 0.5).float()
                loss = F.cross_entropy(model(x), y)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
            h = evaluate(model, test_loader, device, binarize=True, hard=True, flat=True)
            print(f"  Client {i}: {100*h:.1f}%")
    else:
        client_models = []
        for i in range(K):
            m = copy.deepcopy(global_model)
            path = os.path.join(args.checkpoint_dir, f'client_{i}.pt')
            m.load_state_dict(torch.load(path, map_location=device))
            client_models.append(m)
            print(f"  Loaded client {i} from {path}")

    # Run BFS
    print("\nRunning BFS...")
    acc, selected = boolean_feature_selection(
        client_models, probe_loader, test_loader, device,
        num_select=args.num_select, binarize=True, flat=True)
    print(f"  BFS accuracy: {100*acc:.1f}%")

    # Export
    print("\nExporting to Verilog...")
    verilog_path = export_bfs_circuit(
        client_models, selected, num_classes=10, tau=10.0,
        output_path=args.output)

    # Try Yosys
    print("\nRunning synthesis...")
    run_yosys_stats(verilog_path)
