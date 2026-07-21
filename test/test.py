import unittest

import jax
from jax import numpy as jnp

from functools import partial

from regularized_lqr_jax.helpers import (
    compute_residual,
    project_psd_cone,
    symmetrize,
    regularize,
)
from regularized_lqr_jax.solver import factor, factor_parallel, solve, solve_parallel
from regularized_lqr_jax.types import (
    FactorizationInputs,
    SolveInputs,
)

jax.config.update("jax_enable_x64", True)


class TestRegularizedLQR(unittest.TestCase):
    def setUp(self):
        n = 4
        m = 2
        T = 30

        self.T = T

        key = jax.random.PRNGKey(0)

        key, subkey = jax.random.split(key)
        self.A = jax.random.uniform(subkey, (T, n, n))

        key, subkey = jax.random.split(key)
        self.B = jax.random.uniform(subkey, (T, n, m))

        key, subkey = jax.random.split(key)
        self.c = jax.random.uniform(subkey, (T + 1, n))

        key, subkey = jax.random.split(key)
        self.Q = jax.random.uniform(subkey, (T + 1, n, n))
        self.Q = jax.vmap(symmetrize)(self.Q)

        key, subkey = jax.random.split(key)
        self.M = jax.random.uniform(subkey, (T, n, m))

        key, subkey = jax.random.split(key)
        self.R = jax.random.uniform(subkey, (T, m, m))
        self.R = jax.vmap(symmetrize)(self.R)

        self.Q, self.R = regularize(Q=self.Q, R=self.R, M=self.M, psd_delta=1e-3)

        key, subkey = jax.random.split(key)
        self.q = jax.random.uniform(subkey, (T + 1, n))

        key, subkey = jax.random.split(key)
        self.r = jax.random.uniform(subkey, (T, m))

        key, subkey = jax.random.split(key)
        self.Δ = jnp.abs(jax.random.uniform(subkey, (T + 1, n, n)))
        self.Δ = jax.vmap(partial(project_psd_cone, delta=1e-3))(self.Δ)
        self.Δ_L = jax.vmap(jnp.linalg.cholesky)(self.Δ)

    def test(self):
        for use_parallel_method in [False, True]:
            with self.subTest(use_parallel_method=use_parallel_method):
                factor_method = factor_parallel if use_parallel_method else factor
                solve_method = solve_parallel if use_parallel_method else solve

                factorization_inputs = FactorizationInputs(
                    A=self.A,
                    B=self.B,
                    Q=self.Q,
                    M=self.M,
                    R=self.R,
                    Δ_L=self.Δ_L,
                )

                factorization_outputs = factor_method(factorization_inputs)

                solve_inputs = SolveInputs(
                    self.q,
                    self.r,
                    self.c,
                )

                solve_outputs = solve_method(
                    factorization_inputs,
                    factorization_outputs,
                    solve_inputs,
                )

                residual = compute_residual(
                    factorization_inputs,
                    solve_inputs,
                    solve_outputs,
                )

                if use_parallel_method:
                    self.assertLess(jnp.linalg.norm(residual), 1e-8)
                else:
                    self.assertLess(jnp.linalg.norm(residual), 1e-12)

    def test_singular_delta_factor(self):
        for use_parallel_method in [False, True]:
            for Δ_L in [
                jnp.zeros_like(self.Δ_L),
                jnp.tile(
                    jnp.diag(jnp.array([1.0, 0.0, 0.25, 0.0])),
                    (self.T + 1, 1, 1),
                ),
            ]:
                with self.subTest(use_parallel_method=use_parallel_method):
                    factor_method = factor_parallel if use_parallel_method else factor
                    solve_method = solve_parallel if use_parallel_method else solve

                    factorization_inputs = FactorizationInputs(
                        A=self.A,
                        B=self.B,
                        Q=self.Q,
                        M=self.M,
                        R=self.R,
                        Δ_L=Δ_L,
                    )
                    factorization_outputs = factor_method(factorization_inputs)
                    solve_inputs = SolveInputs(self.q, self.r, self.c)
                    solve_outputs = solve_method(
                        factorization_inputs,
                        factorization_outputs,
                        solve_inputs,
                    )
                    residual = compute_residual(
                        factorization_inputs,
                        solve_inputs,
                        solve_outputs,
                    )

                    self.assertLess(jnp.linalg.norm(residual), 1e-8)

    def test_float32_is_preserved_with_x64_enabled(self):
        n = 4
        m = 2
        T = 12
        key = jax.random.PRNGKey(1)
        key, *subkeys = jax.random.split(key, 7)

        factorization_inputs = FactorizationInputs(
            A=0.15 * jax.random.normal(subkeys[0], (T, n, n), dtype=jnp.float32),
            B=0.2 * jax.random.normal(subkeys[1], (T, n, m), dtype=jnp.float32),
            Q=jnp.tile(2.0 * jnp.eye(n, dtype=jnp.float32), (T + 1, 1, 1)),
            M=jnp.zeros((T, n, m), dtype=jnp.float32),
            R=jnp.tile(2.0 * jnp.eye(m, dtype=jnp.float32), (T, 1, 1)),
            Δ_L=jnp.tile(0.1 * jnp.eye(n, dtype=jnp.float32), (T + 1, 1, 1)),
        )
        solve_inputs = SolveInputs(
            0.1 * jax.random.normal(subkeys[2], (T + 1, n), dtype=jnp.float32),
            0.1 * jax.random.normal(subkeys[3], (T, m), dtype=jnp.float32),
            0.1 * jax.random.normal(subkeys[4], (T + 1, n), dtype=jnp.float32),
        )

        solutions = []
        for factor_method, solve_method in (
            (factor, solve),
            (factor_parallel, solve_parallel),
        ):
            factorization = factor_method(factorization_inputs)
            solution = solve_method(factorization_inputs, factorization, solve_inputs)
            residual = compute_residual(factorization_inputs, solve_inputs, solution)

            for value in jax.tree.leaves((factorization, solution)):
                self.assertEqual(value.dtype, jnp.float32)
            self.assertLess(jnp.max(jnp.abs(residual)), 2e-5)
            solutions.append(solution)

        for sequential, parallel in zip(
            jax.tree.leaves(solutions[0]), jax.tree.leaves(solutions[1])
        ):
            self.assertTrue(jnp.allclose(sequential, parallel, rtol=2e-5, atol=2e-5))


if __name__ == "__main__":
    unittest.main()
