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

    def test(self):
        for use_parallel_method in [False, True]:
            with self.subTest(use_parallel_method=use_parallel_method):
                factor_method = factor_parallel if use_parallel_method else factor
                solve_method = solve_parallel if use_parallel_method else solve

                factorization_inputs = FactorizationInputs(
                    self.A,
                    self.B,
                    self.Q,
                    self.M,
                    self.R,
                    self.Δ,
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


if __name__ == "__main__":
    unittest.main()
