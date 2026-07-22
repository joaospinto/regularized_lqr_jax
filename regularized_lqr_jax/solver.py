"""Compatibility facades for canonical chain-ordered LQR problems.

All numerical work is implemented by :mod:`regularized_lqr_jax.tree_solver`.
These functions only construct the implicit root-at-zero chain plan and select
sequential or parallel execution.
"""

import jax

from regularized_lqr_jax.tree_solver import (
    factor_tree,
    make_chain_lqr_plan,
    solve_tree,
)
from regularized_lqr_jax.types import (
    FactorizationInputs,
    FactorizationOutputs,
    SolveInputs,
    SolveOutputs,
)


def _chain_plan(inputs: FactorizationInputs, *, parallel: bool):
    """Build a static chain plan while tracing a horizon specialization."""
    return make_chain_lqr_plan(inputs.B.shape[0], parallel=parallel)


@jax.jit
def factor(inputs: FactorizationInputs) -> FactorizationOutputs:
    """Factor a chain using the canonical sequential tree implementation."""
    return factor_tree(_chain_plan(inputs, parallel=False), inputs)


@jax.jit
def solve(
    factorization_inputs: FactorizationInputs,
    factorization_outputs: FactorizationOutputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    """Solve a chain RHS using the canonical sequential tree implementation."""
    plan = _chain_plan(factorization_inputs, parallel=False)
    return solve_tree(plan, factorization_inputs, factorization_outputs, solve_inputs)


@jax.jit
def factor_parallel(inputs: FactorizationInputs) -> FactorizationOutputs:
    """Factor a chain using the canonical parallel tree implementation."""
    return factor_tree(_chain_plan(inputs, parallel=True), inputs)


@jax.jit
def solve_parallel(
    factorization_inputs: FactorizationInputs,
    factorization_outputs: FactorizationOutputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    """Solve a chain RHS using the canonical parallel tree implementation."""
    plan = _chain_plan(factorization_inputs, parallel=True)
    return solve_tree(plan, factorization_inputs, factorization_outputs, solve_inputs)
