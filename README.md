This repository provides a JAX implementation of the
algorithms described in https://arxiv.org/abs/2509.16370.

It supports regularized LQR on both chains and arbitrary directed rooted trees:

- `factor` / `solve`: sequential chain algorithm using `jax.lax.scan`;
- `factor_parallel` / `solve_parallel`: parallel chain algorithm using
  associative scans;
- `factor_tree_parallel` / `solve_tree_parallel`: parallel tree algorithm using
  bidirectional rake--compress contraction.

## Parallel tree API

Tree topology setup is an explicit, one-time CPU operation. Factorization is
RHS-independent and reusable, exactly as it is for chains:

```python
from jax_bidirectional_tree_rake_compress import make_tree_contraction_plan
from regularized_lqr_jax.tree_solver import (
    factor_tree_parallel,
    solve_tree_parallel,
)
from regularized_lqr_jax.types import FactorizationInputs, SolveInputs

plan = make_tree_contraction_plan(parents)

# Node arrays Q and Delta_L remain in node order. Edge arrays A, B, M, and R
# describe plan.edge_parents[e] -> plan.edge_children[e].
lhs = FactorizationInputs(A=A, B=B, Q=Q, M=M, R=R, Δ_L=Delta_L)
factorization = factor_tree_parallel(plan, lhs)

rhs = SolveInputs(q=q, r=r, c=c)
solution = solve_tree_parallel(plan, lhs, factorization, rhs)
```

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
