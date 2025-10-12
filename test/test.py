import unittest

from itertools import product

import jax
from jax import numpy as jnp

from regularized_lqr_jax.helpers import compute_residual, symmetrize, regularize
from regularized_lqr_jax.solver import solve, solve_parallel

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

    def test(self):
        test_data = product([0.5, 0.0], [False, True])

        for δ, use_parallel_method in test_data:
            with self.subTest(δ=δ, use_parallel_method=use_parallel_method):
                method = solve_parallel if use_parallel_method else solve

                X, U, Y, P, p, K, k = method(
                    A=self.A,
                    B=self.B,
                    Q=self.Q,
                    M=self.M,
                    R=self.R,
                    q=self.q,
                    r=self.r,
                    c=self.c,
                    δ=δ,
                )

                residual = compute_residual(
                    A=self.A,
                    B=self.B,
                    Q=self.Q,
                    M=self.M,
                    R=self.R,
                    q=self.q,
                    r=self.r,
                    c=self.c,
                    X=X,
                    U=U,
                    Y=Y,
                    δ=δ,
                )

                if use_parallel_method:
                    self.assertLess(jnp.linalg.norm(residual), 1e-3)
                else:
                    self.assertLess(jnp.linalg.norm(residual), 1e-9)


if __name__ == "__main__":
    unittest.main()
