"""Canonical definitions of the 16 boolean gate operations on (a, b) in {0,1}^2.

Each gate is defined by its truth table -- the output value for inputs
(0,0), (0,1), (1,0), (1,1).  Three renderers produce equivalent expressions
in different target formats:

    to_verilog(idx, a, b)  ->  Verilog wire expression  (e.g. "{a} & {b}")
    to_c(idx, a, b)        ->  C bitwise expression for single-sample mode
    to_python(idx)         ->  callable f(a:int, b:int) -> int  (0 or 1)

Single source of truth for everything that emits or simulates difflogic gates.
"""

GATE_NAMES = ['FALSE', 'AND', 'A_GT_B', 'A', 'B_GT_A', 'B', 'XOR', 'OR',
              'NOR', 'XNOR', 'NOT_B', 'A_GEQ_B', 'NOT_A', 'B_GEQ_A', 'NAND', 'TRUE']

# Truth tables: each is (out for ab=00, out for ab=01, out for ab=10, out for ab=11).
GATE_TRUTH = [
    (0, 0, 0, 0),  # 0  FALSE
    (0, 0, 0, 1),  # 1  AND
    (0, 0, 1, 0),  # 2  A > B  = A AND NOT B
    (0, 0, 1, 1),  # 3  A
    (0, 1, 0, 0),  # 4  B > A  = B AND NOT A
    (0, 1, 0, 1),  # 5  B
    (0, 1, 1, 0),  # 6  XOR
    (0, 1, 1, 1),  # 7  OR
    (1, 0, 0, 0),  # 8  NOR
    (1, 0, 0, 1),  # 9  XNOR
    (1, 0, 1, 0),  # 10 NOT B
    (1, 0, 1, 1),  # 11 A >= B
    (1, 1, 0, 0),  # 12 NOT A
    (1, 1, 0, 1),  # 13 B >= A
    (1, 1, 1, 0),  # 14 NAND
    (1, 1, 1, 1),  # 15 TRUE
]

# Verilog expression templates.  {a}, {b} are placeholders for fan-in wire names.
_VERILOG_EXPRS = [
    "1'b0",
    "{a} & {b}",
    "{a} & ~{b}",
    "{a}",
    "~{a} & {b}",
    "{b}",
    "{a} ^ {b}",
    "{a} | {b}",
    "~({a} | {b})",
    "~({a} ^ {b})",
    "~{b}",
    "{a} | ~{b}",
    "~{a}",
    "~{a} | {b}",
    "~({a} & {b})",
    "1'b1",
]

# C expression templates for single-sample mode (each variable holds 0 or 1
# in a long long).  We use (1 ^ x) instead of ~x for NOT so that the result
# stays 1-bit; using ~x would flip all 64 bits and give -1 for 0.
_C_EXPRS = [
    "0",
    "({a} & {b})",
    "({a} & (1 ^ {b}))",
    "{a}",
    "((1 ^ {a}) & {b})",
    "{b}",
    "({a} ^ {b})",
    "({a} | {b})",
    "(1 ^ ({a} | {b}))",
    "(1 ^ ({a} ^ {b}))",
    "(1 ^ {b})",
    "((1 ^ {b}) | {a})",
    "(1 ^ {a})",
    "((1 ^ {a}) | {b})",
    "(1 ^ ({a} & {b}))",
    "1",
]


def to_verilog(idx, a, b):
    """Return the Verilog expression for gate idx with fan-in wires a, b."""
    return _VERILOG_EXPRS[idx].format(a=a, b=b)


def to_c(idx, a, b):
    """Return the C bitwise expression for gate idx with fan-in vars a, b."""
    return _C_EXPRS[idx].format(a=a, b=b)


def to_python(idx):
    """Return f(a:int, b:int) -> int implementing gate idx in pure Python."""
    tt = GATE_TRUTH[idx]
    return lambda a, b: tt[(a << 1) | b]


# --- self-test ---------------------------------------------------------------

def _eval_c_template(expr_template, a, b):
    """Evaluate a C expression template with literal 0/1 values for a, b."""
    expr = expr_template.format(a=str(a), b=str(b))
    expr = expr.replace('&', ' and ').replace('|', ' or ').replace('^', '!=')
    expr = expr.replace(' and  and ', ' and ').replace(' or  or ', ' or ')
    return int(eval(expr))


def _self_test():
    """Verify all three renderers agree with the truth tables."""
    for g in range(16):
        py = to_python(g)
        for ab in range(4):
            a = (ab >> 1) & 1
            b = ab & 1
            expected = GATE_TRUTH[g][ab]
            got_py = py(a, b)
            assert got_py == expected, f'python gate {g} ({a},{b}): got {got_py}, want {expected}'
            got_c = _eval_c_template(_C_EXPRS[g], a, b)
            assert got_c == expected, f'C gate {g} ({a},{b}): got {got_c}, want {expected}'
    return True


if __name__ == '__main__':
    _self_test()
    print(f'All {len(GATE_TRUTH)} gates self-consistent (Python and C templates match truth tables).')
