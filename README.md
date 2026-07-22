This repository provides a JAX implementation of the
algorithms described in https://arxiv.org/abs/2509.16370.

It supports regularized LQR on both chains and arbitrary directed rooted trees:

- `factor` / `solve`: sequential chain facades using `jax.lax.scan`;
- `factor_parallel` / `solve_parallel`: parallel chain facades using
  `jax.lax.associative_scan`;
- `factor_tree` / `solve_tree`: rooted-tree algorithms whose reusable plan
  selects sequential or parallel contraction.

## Tree API

Tree topology setup is an explicit, one-time CPU operation. Factorization is
RHS-independent and reusable, exactly as it is for chains:

```python
from regularized_lqr_jax.tree_solver import (
    factor_tree,
    make_tree_lqr_plan,
    solve_tree,
)
from regularized_lqr_jax.types import FactorizationInputs, SolveInputs

plan = make_tree_lqr_plan(parents, parallel=True)

# Node arrays Q and Delta_L remain in node order. Edge arrays A, B, M, and R
# describe plan.edge_parents[e] -> plan.edge_children[e].
lhs = FactorizationInputs(A=A, B=B, Q=Q, M=M, R=R, Δ_L=Delta_L)
factorization = factor_tree(plan, lhs)

rhs = SolveInputs(q=q, r=r, c=c)
solution = solve_tree(plan, lhs, factorization, rhs)
```

`make_tree_lqr_plan` selects `jax.lax.scan` for sequential chains,
`jax.lax.associative_scan` for parallel chains, and unrolled rake-only or
rake--compress contraction for branching trees. The topology and this selection
are computed once, outside the compiled factor and solve calls.

The chain entry points contain no independent numerical solver. They create the
canonical root-at-zero chain plan from the static horizon while tracing, then
delegate to `factor_tree` and `solve_tree`.

The original `factor_tree_parallel`, `solve_tree_parallel`, and
`factor_and_solve_tree_parallel` names remain as compatibility aliases. Their
behavior is determined by the plan, despite the historical suffix.

The tree solver uses
[`jax-bidirectional-tree-rake-compress`](https://github.com/joaospinto/jax-bidirectional-tree-rake-compress)
for quadratic subtree factorization/recovery, upward affine RHS recovery, and
downward state expansion. See
[the algorithm note](docs/tree_regularized_lqr.md) for the local contraction
rules and the relationship to the sequential and associative-scan chain paths.

## Verification and benchmarks

```sh
uv run python -m unittest discover -v
uv run python benchmarks/bench_tree_regularized_lqr.py
```

The benchmark pretty-prints setup, factor, solve, factor+solve, and
setup+factor+solve timings separately for chain, balanced, comb, and star
topologies.
