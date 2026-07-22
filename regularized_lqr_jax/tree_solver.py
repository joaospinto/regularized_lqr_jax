"""Dual-regularized LQR on directed rooted trees.

Call :func:`make_tree_lqr_plan` once on the CPU, order edge data using
``plan.edge_children``, and reuse the plan for every factorization and RHS. The
plan selects sequential or parallel execution; the numerical factorization and
solve code is shared.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jax_bidirectional_tree_rake_compress import (
    ContractionExecutor,
    ContractionSchedule,
    TreeContractionPlan,
    make_tree_contraction_plan,
    tree_contract,
    tree_expand,
)
from jax.typing import ArrayLike

from regularized_lqr_jax.helpers import (
    compose_value_functions,
    factor_feedback,
    factor_value_node,
    form_delta,
    make_edge_value,
    solve_feedforward,
    stable_F_solve,
    symmetrize,
    terminalize_value,
)
from regularized_lqr_jax.types import (
    FactorizationOutputs,
    FactorizationInputs,
    SolveInputs,
    SolveOutputs,
)


def make_tree_lqr_plan(
    parents: ArrayLike,
    *,
    parallel: bool,
    root: int | None = None,
) -> TreeContractionPlan:
    """Preprocess a tree once and select its LQR execution strategy.

    Chains use ``lax.scan`` sequentially and ``lax.associative_scan`` in
    parallel. Branching trees use the unrolled executor with rake-only or
    rake--compress scheduling respectively.
    """
    schedule = (
        ContractionSchedule.RAKE_COMPRESS if parallel else ContractionSchedule.RAKE_ONLY
    )

    return make_tree_contraction_plan(
        parents,
        root=root,
        schedule=schedule,
        executor=ContractionExecutor.AUTO,
    )


def make_chain_lqr_plan(
    num_edges: int,
    *,
    parallel: bool,
) -> TreeContractionPlan:
    """Create the canonical root-at-zero chain plan for one horizon."""
    if num_edges < 0:
        raise ValueError("num_edges must be nonnegative")
    return make_tree_lqr_plan([-1, *range(num_edges)], parallel=parallel)


@dataclass(frozen=True)
class _QuadraticValueAlgebra:
    """Rake--compress algebra recovering every subtree value ``P_i``."""

    def rake(self, path, leaf):
        return terminalize_value(path, leaf), leaf

    def combine_branches(self, left, right):
        return symmetrize(left + right)

    def absorb_branch(self, node, message):
        return symmetrize(node + message)

    def compress(self, left, middle, right):
        augmented_right = (right[0], right[1], symmetrize(right[2] + middle))
        return compose_value_functions(left, augmented_right), augmented_right

    def expand_compress(self, residual, parent_output, child_output):
        del parent_output
        return terminalize_value(residual, child_output)

    def expand_rake(self, residual, parent_output):
        del parent_output
        return residual


@dataclass(frozen=True)
class _RakeOnlyQuadraticValueAlgebra:
    """Riccati-stable value elimination for schedules without compression."""

    def rake(self, path, leaf):
        A, B, M, R, Delta_L = path
        W, _ = factor_value_node(Delta_L, leaf)
        K, _, H = factor_feedback(A, B, M, R, W)
        return symmetrize(A.T @ W @ A + K.T @ H), leaf

    def combine_branches(self, left, right):
        return symmetrize(left + right)

    def absorb_branch(self, node, message):
        return symmetrize(node + message)

    def compress(self, left, middle, right):
        del left, middle, right
        raise RuntimeError("rake-only algebra received a compression")

    def expand_compress(self, residual, parent_output, child_output):
        del residual, parent_output, child_output
        raise RuntimeError("rake-only algebra received a compression residual")

    def expand_rake(self, residual, parent_output):
        del parent_output
        return residual


@dataclass(frozen=True)
class _UpwardAffineAlgebra:
    """Recover ``p_i = q_i + sum_e (Z_e p_child + z_e)`` in parallel."""

    def rake(self, path, leaf):
        Z, z = path
        return Z @ leaf + z, leaf

    def combine_branches(self, left, right):
        return left + right

    def absorb_branch(self, node, message):
        return node + message

    def compress(self, left, middle, right):
        Z_left, z_left = left
        Z_right, z_right = right
        right_offset = middle + z_right
        return (
            Z_left @ Z_right,
            z_left + Z_left @ right_offset,
        ), (Z_right, right_offset)

    def expand_compress(self, residual, parent_output, child_output):
        del parent_output
        Z_right, right_offset = residual
        return Z_right @ child_output + right_offset

    def expand_rake(self, residual, parent_output):
        del parent_output
        return residual


@dataclass(frozen=True)
class _RakeOnlyUpwardAffineAlgebra:
    """Stable affine Riccati recurrence for schedules without compression."""

    def rake(self, path, leaf):
        A, B, r, c_child, Delta_child, W_child, G_cho, K = path
        f_child = Delta_child @ leaf - c_child
        g_child = leaf - W_child @ f_child
        h = r + B.T @ g_child
        return A.T @ g_child + K.T @ h, leaf

    def combine_branches(self, left, right):
        return left + right

    def absorb_branch(self, node, message):
        return node + message

    def compress(self, left, middle, right):
        del left, middle, right
        raise RuntimeError("rake-only algebra received a compression")

    def expand_compress(self, residual, parent_output, child_output):
        del residual, parent_output, child_output
        raise RuntimeError("rake-only algebra received a compression residual")

    def expand_rake(self, residual, parent_output):
        del parent_output
        return residual


@dataclass(frozen=True)
class _DownwardAffineAlgebra:
    """Broadcast affine closed-loop dynamics from the root to every node."""

    def rake(self, path, leaf):
        return jnp.zeros_like(leaf), path

    def combine_branches(self, left, right):
        return left + right

    def absorb_branch(self, node, message):
        return node + message

    def compress(self, left, middle, right):
        del middle
        T_left, b_left = left
        T_right, b_right = right
        return (
            T_right @ T_left,
            T_right @ b_left + b_right,
        ), left

    def expand_compress(self, residual, parent_output, child_output):
        del child_output
        T_left, b_left = residual
        return T_left @ parent_output + b_left

    def expand_rake(self, residual, parent_output):
        T, b = residual
        return T @ parent_output + b


@dataclass(frozen=True)
class _RakeOnlyDownwardAlgebra:
    """Stable forward dynamics recurrence for schedules without compression."""

    def rake(self, path, leaf):
        return jnp.zeros_like(leaf), path

    def combine_branches(self, left, right):
        return left + right

    def absorb_branch(self, node, message):
        return node + message

    def compress(self, left, middle, right):
        del left, middle, right
        raise RuntimeError("rake-only algebra received a compression")

    def expand_compress(self, residual, parent_output, child_output):
        del residual, parent_output, child_output
        raise RuntimeError("rake-only algebra received a compression residual")

    def expand_rake(self, residual, parent_output):
        A, B, K, k, f_child, S_cho, Delta_L, P = residual
        u = K @ parent_output + k
        return stable_F_solve(S_cho, Delta_L, P, A @ parent_output + B @ u - f_child)


def _uses_compression(plan: TreeContractionPlan) -> bool:
    """Return a static indication that this plan can execute compression."""
    if plan.executor is ContractionExecutor.ASSOCIATIVE_SCAN:
        return True
    if plan.executor is ContractionExecutor.SCAN:
        return False
    return any(round_.compressions.shape[0] for round_ in plan.rounds)


def _compositional_factorization_data(
    outputs: FactorizationOutputs,
) -> tuple[jax.Array, jax.Array]:
    """Return the fields required by plans that perform compression."""
    if outputs.ApBK is None or outputs.F_inv_ApBK is None:
        raise ValueError(
            "compression plans require compositional factorization data"
        )
    return outputs.ApBK, outputs.F_inv_ApBK


def _check_factor_shapes(
    plan: TreeContractionPlan, inputs: FactorizationInputs
) -> None:
    if inputs.Q.shape[0] != plan.num_nodes or inputs.Δ_L.shape[0] != plan.num_nodes:
        raise ValueError("Q and Δ_L must have one block per plan node")
    for name in ("A", "B", "M", "R"):
        if getattr(inputs, name).shape[0] != plan.num_edges:
            raise ValueError(f"{name} must have one block per plan edge")


def _check_solve_shapes(plan: TreeContractionPlan, inputs: SolveInputs) -> None:
    if inputs.q.shape[0] != plan.num_nodes or inputs.c.shape[0] != plan.num_nodes:
        raise ValueError("q and c must have one vector per plan node")
    if inputs.r.shape[0] != plan.num_edges:
        raise ValueError("r must have one vector per plan edge")


@jax.jit
def factor_tree(
    plan: TreeContractionPlan,
    inputs: FactorizationInputs,
) -> FactorizationOutputs:
    """Factor a regularized tree LQR problem using its planned executor.

    ``plan`` is topology-only CPU preprocessing. Node data stays in original
    node order; edge data follows ``plan.edge_children``.
    """
    _check_factor_shapes(plan, inputs)
    uses_compression = _uses_compression(plan)
    Delta = jax.vmap(form_delta)(inputs.Δ_L)
    if uses_compression:
        edge_values = jax.vmap(make_edge_value)(
            inputs.A,
            inputs.B,
            inputs.M,
            inputs.R,
            Delta[plan.edge_children],
        )
        value_algebra = _QuadraticValueAlgebra()
    else:
        edge_values = (
            inputs.A,
            inputs.B,
            inputs.M,
            inputs.R,
            inputs.Δ_L[plan.edge_children],
        )
        value_algebra = _RakeOnlyQuadraticValueAlgebra()
    root_P, value_tape = tree_contract(plan, inputs.Q, edge_values, value_algebra)
    P = tree_expand(plan, value_tape, root_P, value_algebra)
    P = symmetrize(P)

    W, S_cho = jax.vmap(factor_value_node)(inputs.Δ_L, P)
    children = plan.edge_children
    K, G_cho, _ = jax.vmap(factor_feedback)(
        inputs.A, inputs.B, inputs.M, inputs.R, W[children]
    )
    if uses_compression:
        ApBK = jax.vmap(lambda A, B, K: A + B @ K)(inputs.A, inputs.B, K)
        F_inv_ApBK = jax.vmap(stable_F_solve)(
            S_cho[children], inputs.Δ_L[children], P[children], ApBK
        )
    else:
        ApBK = None
        F_inv_ApBK = None
    return FactorizationOutputs(
        P=P,
        K=K,
        W=W,
        G_cho=G_cho,
        S_cho=S_cho,
        ApBK=ApBK,
        F_inv_ApBK=F_inv_ApBK,
    )


@jax.jit
def solve_tree(
    plan: TreeContractionPlan,
    factorization_inputs: FactorizationInputs,
    factorization_outputs: FactorizationOutputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    """Solve one RHS using a reusable tree factorization."""
    _check_factor_shapes(plan, factorization_inputs)
    _check_solve_shapes(plan, solve_inputs)
    inputs = factorization_inputs
    outputs = factorization_outputs
    parents, children = plan.edge_parents, plan.edge_children
    uses_compression = _uses_compression(plan)
    Delta = jax.vmap(form_delta)(inputs.Δ_L)

    if uses_compression:
        ApBK, F_inv_ApBK = _compositional_factorization_data(outputs)
        Z = jnp.swapaxes(F_inv_ApBK, -2, -1)
        z = jax.vmap(lambda K, r, Acl, W, c: K.T @ r + Acl.T @ (W @ c))(
            outputs.K,
            solve_inputs.r,
            ApBK,
            outputs.W[children],
            solve_inputs.c[children],
        )
        upward_paths = (Z, z)
        upward_algebra = _UpwardAffineAlgebra()
    else:
        upward_paths = (
            inputs.A,
            inputs.B,
            solve_inputs.r,
            solve_inputs.c[children],
            Delta[children],
            outputs.W[children],
            outputs.G_cho,
            outputs.K,
        )
        upward_algebra = _RakeOnlyUpwardAffineAlgebra()
    root_p, affine_tape = tree_contract(
        plan, solve_inputs.q, upward_paths, upward_algebra
    )
    p = tree_expand(plan, affine_tape, root_p, upward_algebra)
    f = jax.vmap(lambda Delta_i, p_i, c_i: Delta_i @ p_i - c_i)(
        Delta, p, solve_inputs.c
    )
    k = jax.vmap(solve_feedforward)(
        inputs.B,
        solve_inputs.r,
        outputs.W[children],
        outputs.G_cho,
        f[children],
        p[children],
    )

    root = plan.root
    x_root = -stable_F_solve(
        outputs.S_cho[root], inputs.Δ_L[root], outputs.P[root], f[root]
    )
    if uses_compression:
        b = jax.vmap(
            lambda S_cho, Delta_L, P, B, k_i, f_child: stable_F_solve(
                S_cho, Delta_L, P, B @ k_i - f_child
            )
        )(
            outputs.S_cho[children],
            inputs.Δ_L[children],
            outputs.P[children],
            inputs.B,
            k,
            f[children],
        )
        downward_paths = (F_inv_ApBK, b)
        downward_algebra = _DownwardAffineAlgebra()
    else:
        downward_paths = (
            inputs.A,
            inputs.B,
            outputs.K,
            k,
            f[children],
            outputs.S_cho[children],
            inputs.Δ_L[children],
            outputs.P[children],
        )
        downward_algebra = _RakeOnlyDownwardAlgebra()
    dummy_nodes = jnp.zeros((plan.num_nodes,), dtype=inputs.Q.dtype)
    _, state_tape = tree_contract(
        plan,
        dummy_nodes,
        downward_paths,
        downward_algebra,
    )
    X = tree_expand(plan, state_tape, x_root, downward_algebra)
    U = jax.vmap(lambda K, x_parent, k_i: K @ x_parent + k_i)(outputs.K, X[parents], k)
    Y = jax.vmap(lambda P_i, x_i, p_i: P_i @ x_i + p_i)(outputs.P, X, p)
    return SolveOutputs(X=X, U=U, Y=Y, p=p, k=k)


@jax.jit
def factor_and_solve_tree(
    plan: TreeContractionPlan,
    factorization_inputs: FactorizationInputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    """Convenience factor-and-solve entry point for one tree RHS."""
    factorization_outputs = factor_tree(plan, factorization_inputs)
    return solve_tree(plan, factorization_inputs, factorization_outputs, solve_inputs)


@jax.jit
def compute_tree_residual(
    plan: TreeContractionPlan,
    factorization_inputs: FactorizationInputs,
    solve_inputs: SolveInputs,
    solve_outputs: SolveOutputs,
) -> jax.Array:
    """Return the flattened regularized KKT residual for a tree solution."""
    _check_factor_shapes(plan, factorization_inputs)
    _check_solve_shapes(plan, solve_inputs)
    inputs = factorization_inputs
    parents, children = plan.edge_parents, plan.edge_children
    Delta = jax.vmap(form_delta)(inputs.Δ_L)
    X, U, Y = solve_outputs.X, solve_outputs.U, solve_outputs.Y

    state_stationarity = jax.vmap(lambda Q, x, q, y: Q @ x + q - y)(
        inputs.Q, X, solve_inputs.q, Y
    )
    outgoing = jax.vmap(lambda M, u, A, y: M @ u + A.T @ y)(
        inputs.M, U, inputs.A, Y[children]
    )
    state_stationarity = state_stationarity.at[parents].add(outgoing)
    control_stationarity = jax.vmap(
        lambda M, x, R, u, B, y, r: M.T @ x + R @ u + B.T @ y + r
    )(
        inputs.M,
        X[parents],
        inputs.R,
        U,
        inputs.B,
        Y[children],
        solve_inputs.r,
    )
    root = plan.root
    root_dynamics = -X[root] - Delta[root] @ Y[root] + solve_inputs.c[root]
    edge_dynamics = jax.vmap(
        lambda A, xp, B, u, xc, c, Delta_i, y: A @ xp + B @ u - xc + c - Delta_i @ y
    )(
        inputs.A,
        X[parents],
        inputs.B,
        U,
        X[children],
        solve_inputs.c[children],
        Delta[children],
        Y[children],
    )
    return jnp.concatenate(
        [
            state_stationarity.reshape(-1),
            control_stationarity.reshape(-1),
            root_dynamics.reshape(-1),
            edge_dynamics.reshape(-1),
        ]
    )


# Compatibility aliases for the original parallel-tree API. The plan now
# determines execution, so the neutral names above are canonical.
factor_tree_parallel = factor_tree
solve_tree_parallel = solve_tree
factor_and_solve_tree_parallel = factor_and_solve_tree
