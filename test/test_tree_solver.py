"""Correctness tests for parallel regularized LQR on rooted trees."""

from __future__ import annotations

import unittest

import jax
import jax.numpy as jnp
import numpy as np
from jax_bidirectional_tree_rake_compress import make_tree_contraction_plan

from regularized_lqr_jax.solver import factor_parallel, solve_parallel
from regularized_lqr_jax.tree_solver import (
    compute_tree_residual,
    factor_and_solve_tree_parallel,
    factor_tree_parallel,
    solve_tree_parallel,
)
from regularized_lqr_jax.types import FactorizationInputs, SolveInputs

jax.config.update("jax_enable_x64", True)


def _random_problem(parents, seed=0, n=3, m=2, regularized=True):
    plan = make_tree_contraction_plan(parents)
    rng = np.random.default_rng(seed)
    V, E = plan.num_nodes, plan.num_edges
    A = 0.35 * rng.standard_normal((E, n, n))
    B = 0.5 * rng.standard_normal((E, n, m))
    M = 0.08 * rng.standard_normal((E, n, m))
    R = (
        np.stack(
            [
                matrix @ matrix.T + 2.0 * np.eye(m)
                for matrix in rng.normal(size=(E, m, m))
            ]
        )
        if E
        else np.zeros((0, m, m))
    )
    Q = np.tile(2.0 * np.eye(n)[None], (V, 1, 1))
    for edge, parent in enumerate(np.asarray(plan.edge_parents)):
        Q[parent] += M[edge] @ np.linalg.solve(R[edge], M[edge].T)
    Delta_L = (
        np.stack([np.diag(np.sqrt(0.02 + 0.03 * rng.random(n))) for _ in range(V)])
        if regularized
        else np.zeros((V, n, n))
    )
    factor_inputs = FactorizationInputs(
        A=jnp.asarray(A),
        B=jnp.asarray(B),
        Q=jnp.asarray(Q),
        M=jnp.asarray(M),
        R=jnp.asarray(R),
        Δ_L=jnp.asarray(Delta_L),
    )
    solve_inputs = SolveInputs(
        q=jnp.asarray(rng.standard_normal((V, n))),
        r=jnp.asarray(rng.standard_normal((E, m))),
        c=jnp.asarray(rng.standard_normal((V, n))),
    )
    return plan, factor_inputs, solve_inputs


def _dense_kkt_solution(plan, factor_inputs, solve_inputs):
    """Independently assemble and solve the full tree KKT system in NumPy."""
    A = np.asarray(factor_inputs.A)
    B = np.asarray(factor_inputs.B)
    Q = np.asarray(factor_inputs.Q)
    M = np.asarray(factor_inputs.M)
    R = np.asarray(factor_inputs.R)
    Delta_L = np.asarray(factor_inputs.Δ_L)
    Delta = Delta_L @ np.swapaxes(Delta_L, -2, -1)
    q = np.asarray(solve_inputs.q)
    r = np.asarray(solve_inputs.r)
    c = np.asarray(solve_inputs.c)
    parents = np.asarray(plan.edge_parents)
    children = np.asarray(plan.edge_children)

    V, E = plan.num_nodes, plan.num_edges
    n, m = Q.shape[-1], R.shape[-1]
    num_x = V * n
    num_u = E * m
    num_primal = num_x + num_u
    size = num_primal + V * n
    matrix = np.zeros((size, size), dtype=Q.dtype)
    rhs = np.zeros(size, dtype=Q.dtype)

    def x_slice(node):
        return slice(node * n, (node + 1) * n)

    def u_slice(edge):
        start = num_x + edge * m
        return slice(start, start + m)

    def y_slice(node):
        start = num_primal + node * n
        return slice(start, start + n)

    for node in range(V):
        xs = x_slice(node)
        matrix[xs, xs] = Q[node]
        rhs[xs] = -q[node]

    for edge, parent in enumerate(parents):
        xs, us = x_slice(int(parent)), u_slice(edge)
        matrix[xs, us] = M[edge]
        matrix[us, xs] = M[edge].T
        matrix[us, us] = R[edge]
        rhs[us] = -r[edge]

    root = int(plan.root)
    xs, ys = x_slice(root), y_slice(root)
    matrix[ys, xs] = -np.eye(n)
    matrix[xs, ys] = -np.eye(n)
    matrix[ys, ys] = -Delta[root]
    rhs[ys] = -c[root]

    for edge, (parent, child) in enumerate(zip(parents, children, strict=True)):
        parent, child = int(parent), int(child)
        x_parent = x_slice(parent)
        x_child = x_slice(child)
        us = u_slice(edge)
        ys = y_slice(child)
        matrix[ys, x_parent] = A[edge]
        matrix[x_parent, ys] = A[edge].T
        matrix[ys, us] = B[edge]
        matrix[us, ys] = B[edge].T
        matrix[ys, x_child] = -np.eye(n)
        matrix[x_child, ys] = -np.eye(n)
        matrix[ys, ys] = -Delta[child]
        rhs[ys] = -c[child]

    dense = np.linalg.solve(matrix, rhs)
    np.testing.assert_allclose(matrix @ dense, rhs, atol=2e-10, rtol=2e-10)
    return (
        dense[:num_x].reshape(V, n),
        dense[num_x:num_primal].reshape(E, m),
        dense[num_primal:].reshape(V, n),
    )


def _random_permuted_parents(rng, num_nodes):
    """Generate a random recursive tree and then destroy topological order."""
    ordered = np.array(
        [-1, *(rng.integers(0, child) for child in range(1, num_nodes))],
        dtype=np.int32,
    )
    permutation = rng.permutation(num_nodes)
    inverse = np.empty(num_nodes, dtype=np.int32)
    inverse[permutation] = np.arange(num_nodes, dtype=np.int32)
    parents = np.full(num_nodes, -1, dtype=np.int32)
    for old_child in range(1, num_nodes):
        parents[inverse[old_child]] = inverse[ordered[old_child]]
    return parents


class TestTreeSolver(unittest.TestCase):
    def test_tree_solution_has_small_kkt_residual(self):
        parent_arrays = (
            [-1],
            [-1, 0],
            [-1, 0, 1, 2, 3, 4, 5, 6],
            [-1, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 1, 1, 3, 2, 6, 6, 8, 3],
            [4, 4, 0, 1, -1, 1, 5, 2, 2],
        )
        for parents in parent_arrays:
            for regularized in (False, True):
                with self.subTest(parents=parents, regularized=regularized):
                    plan, factor_inputs, solve_inputs = _random_problem(
                        parents, seed=len(parents), regularized=regularized
                    )
                    solution = factor_and_solve_tree_parallel(
                        plan, factor_inputs, solve_inputs
                    )
                    residual = compute_tree_residual(
                        plan, factor_inputs, solve_inputs, solution
                    )
                    np.testing.assert_allclose(residual, 0.0, atol=2e-9, rtol=2e-9)

    def test_chain_matches_existing_associative_scan(self):
        num_nodes = 9
        for regularized in (False, True):
            with self.subTest(regularized=regularized):
                plan, factor_inputs, solve_inputs = _random_problem(
                    [-1, *range(num_nodes - 1)],
                    seed=31,
                    n=4,
                    m=2,
                    regularized=regularized,
                )
                tree_solution = factor_and_solve_tree_parallel(
                    plan, factor_inputs, solve_inputs
                )
                chain_factorization = factor_parallel(factor_inputs)
                chain_solution = solve_parallel(
                    factor_inputs, chain_factorization, solve_inputs
                )
                np.testing.assert_allclose(
                    tree_solution.X, chain_solution.X, atol=2e-9, rtol=2e-9
                )
                np.testing.assert_allclose(
                    tree_solution.U, chain_solution.U, atol=2e-9, rtol=2e-9
                )
                np.testing.assert_allclose(
                    tree_solution.Y, chain_solution.Y, atol=2e-9, rtol=2e-9
                )

    def test_tree_solution_matches_independent_dense_kkt(self):
        parent_arrays = (
            [-1],
            [-1, 0, 1, 2, 3, 4, 5, 6],
            [-1, 0, 0, 0, 0, 0, 0, 0],
            [-1, 0, 0, 1, 1, 3, 2, 6, 6, 8, 3],
            [4, 4, 0, 1, -1, 1, 5, 2, 2],
        )
        for parents in parent_arrays:
            for regularized in (False, True):
                with self.subTest(parents=parents, regularized=regularized):
                    plan, factor_inputs, solve_inputs = _random_problem(
                        parents,
                        seed=101 + len(parents),
                        n=3,
                        m=2,
                        regularized=regularized,
                    )
                    solution = factor_and_solve_tree_parallel(
                        plan, factor_inputs, solve_inputs
                    )
                    dense_X, dense_U, dense_Y = _dense_kkt_solution(
                        plan, factor_inputs, solve_inputs
                    )
                    np.testing.assert_allclose(
                        solution.X, dense_X, atol=3e-9, rtol=3e-9
                    )
                    np.testing.assert_allclose(
                        solution.U, dense_U, atol=3e-9, rtol=3e-9
                    )
                    np.testing.assert_allclose(
                        solution.Y, dense_Y, atol=3e-9, rtol=3e-9
                    )

    def test_random_permuted_trees_match_dense_kkt(self):
        for seed in range(16):
            with self.subTest(seed=seed):
                rng = np.random.default_rng(10_000 + seed)
                num_nodes = 2 + (7 * seed) % 16
                parents = _random_permuted_parents(rng, num_nodes)
                n = 1 + seed % 4
                m = 1 + (seed // 4) % 3
                regularized = bool(seed % 2)
                plan, factor_inputs, solve_inputs = _random_problem(
                    parents,
                    seed=20_000 + seed,
                    n=n,
                    m=m,
                    regularized=regularized,
                )
                solution = factor_and_solve_tree_parallel(
                    plan, factor_inputs, solve_inputs
                )
                dense_X, dense_U, dense_Y = _dense_kkt_solution(
                    plan, factor_inputs, solve_inputs
                )
                np.testing.assert_allclose(solution.X, dense_X, atol=8e-9, rtol=8e-9)
                np.testing.assert_allclose(solution.U, dense_U, atol=8e-9, rtol=8e-9)
                np.testing.assert_allclose(solution.Y, dense_Y, atol=8e-9, rtol=8e-9)

                residual = compute_tree_residual(
                    plan, factor_inputs, solve_inputs, solution
                )
                np.testing.assert_allclose(residual, 0.0, atol=8e-9, rtol=8e-9)

    def test_factorization_is_reusable_across_rhs(self):
        plan, factor_inputs, solve_inputs = _random_problem(
            [-1, 0, 0, 1, 1, 2, 2, 5, 5, 8, 3], seed=8
        )
        factorization = factor_tree_parallel(plan, factor_inputs)

        for scale in (1.0, -0.4):
            with self.subTest(scale=scale):
                rhs = SolveInputs(
                    q=scale * solve_inputs.q,
                    r=scale * solve_inputs.r,
                    c=scale * solve_inputs.c,
                )
                solution = solve_tree_parallel(plan, factor_inputs, factorization, rhs)
                residual = compute_tree_residual(plan, factor_inputs, rhs, solution)
                np.testing.assert_allclose(residual, 0.0, atol=2e-9, rtol=2e-9)

    def test_tree_solve_is_differentiable_through_rhs(self):
        plan, factor_inputs, solve_inputs = _random_problem(
            [-1, 0, 0, 1, 1, 2, 5], seed=17, n=2, m=1
        )
        factorization = factor_tree_parallel(plan, factor_inputs)

        def objective(q):
            rhs = SolveInputs(q=q, r=solve_inputs.r, c=solve_inputs.c)
            solution = solve_tree_parallel(plan, factor_inputs, factorization, rhs)
            return jnp.sum(solution.X**2) + jnp.sum(solution.U**2)

        gradient = jax.jit(jax.grad(objective))(solve_inputs.q)
        self.assertEqual(gradient.shape, solve_inputs.q.shape)
        self.assertTrue(bool(jnp.all(jnp.isfinite(gradient))))

    def test_float32_tree_residual(self):
        plan, factor_inputs, solve_inputs = _random_problem(
            [-1, 0, 0, 1, 1, 2, 5, 5, 7, 3, 3, 10, 10],
            seed=44,
            n=4,
            m=2,
        )
        factor_inputs = jax.tree.map(
            lambda value: value.astype(jnp.float32), factor_inputs
        )
        solve_inputs = jax.tree.map(
            lambda value: value.astype(jnp.float32), solve_inputs
        )
        solution = factor_and_solve_tree_parallel(plan, factor_inputs, solve_inputs)
        residual = compute_tree_residual(plan, factor_inputs, solve_inputs, solution)
        np.testing.assert_allclose(residual, 0.0, atol=2e-4, rtol=2e-4)

    def test_shape_mismatch_is_reported(self):
        plan, factor_inputs, _ = _random_problem([-1, 0, 0], seed=4)
        bad_inputs = FactorizationInputs(
            A=factor_inputs.A[:1],
            B=factor_inputs.B[:1],
            Q=factor_inputs.Q,
            M=factor_inputs.M[:1],
            R=factor_inputs.R[:1],
            Δ_L=factor_inputs.Δ_L,
        )
        with self.assertRaisesRegex(ValueError, "one block per plan edge"):
            factor_tree_parallel(plan, bad_inputs)


if __name__ == "__main__":
    unittest.main()
