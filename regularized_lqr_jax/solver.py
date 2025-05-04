import jax
from jax import numpy as jnp


from regularized_lqr_jax.helpers import (
    solve_symmetric_positive_definite_system,
    symmetrize,
)


@jax.jit
def solve(
    A: jnp.ndarray,
    B: jnp.ndarray,
    Q: jnp.ndarray,
    M: jnp.ndarray,
    R: jnp.ndarray,
    q: jnp.ndarray,
    r: jnp.ndarray,
    c: jnp.ndarray,
    δ: float,
):
    """
    Solves the following regularized LQR problem:
        [P   C.T] [x] = -[s],
        [C  -δ I] [y]    [c]
    where
        P = diag(P_0, ..., P_N),
        P_i = |-> [Q_i   M_i] if 0 <= i < N
              |   [M_i.T R_i]
              |-> Q_i         if i = N
    and C = [ -I                                                 ]
            [A_1  B_1  -I                                        ]
            [         A_2  B_2  -I                               ]
            [                  A_3  B_3  -I                      ]
            [                               (...)                ]
            [                               A_{N-1}  B_{N-1}  -I ]
    and s = [q[0], r[0], ..., q[N-1], r[N-1], q[N]]
    and k > 0.

    Shapes (where n, m are the state and control dimensions):
        A: [N, n, n],
        B: [N, n, m],
        Q: [N+1, n, n],
        M: [N, n, m],
        R: [N, m, m],
        q: [N+1, n],
        r: [N, m],
        c: [N+1, n],

    Returns:
        X: [N+1, n],
        U: [N, m],
        y: [N+1, n],
        respectively the states, controls, and co-states.
    """
    N, n, m = B.shape

    def reg_lqr_step(V, v, F_inv, Q, M, R, A, B, q, r, c):
        """Performs a single step of the regularized LQR backward pass."""
        I_n = jnp.eye(n)
        W = F_inv @ V
        G = symmetrize(B.T @ W @ B + R)
        g = v + W @ (c - δ * v)
        H = B.T @ W @ A + M.T
        h = r + B.T @ g
        K_k = solve_symmetric_positive_definite_system(
            G, -jnp.hstack((H, h.reshape([-1, 1])))
        )
        K = K_k[:, :-1]
        k = K_k[:, -1]
        V = symmetrize(A.T @ W @ A + Q + K.T @ H)
        v = q + A.T @ g + K.T @ h
        F_inv = jnp.linalg.inv(I_n + δ * V)
        return K, k, V, v, F_inv

    def f(carry, elem):
        """Wraps reg_lqr_step for jax.lax.scan."""
        V, v, F_inv = carry
        i = elem

        K, k, V, v, F_inv = reg_lqr_step(
            V, v, F_inv, Q[i], M[i], R[i], A[i], B[i], q[i], r[i], c[i + 1]
        )

        new_carry = (V, v, F_inv)
        new_output = (K, k, V, v, F_inv)

        return new_carry, new_output

    V_N = Q[N]
    v_N = q[N]
    I_n = jnp.eye(n)
    F_inv_N = jnp.linalg.inv(I_n + δ * V_N)

    K, k, V, v, F_inv = jax.lax.scan(
        f, (V_N, v_N, F_inv_N), jnp.arange(N), N, reverse=True
    )[1]

    V = jnp.concatenate([V, V_N.reshape([1, n, n])])
    v = jnp.concatenate([v, v_N.reshape([1, n])])
    F_inv = jnp.concatenate([F_inv, F_inv_N.reshape([1, n, n])])

    def rollout(K, k, x0, A, B, c, F_inv, v):
        """
        Performs the regularized LQR forward pass.
        """

        T, n, m = B.shape

        def f(carry, elem):
            t = elem

            x = carry
            u = K[t] @ x + k[t]
            f = δ * v[t + 1] - c[t + 1]
            next_x = F_inv[t + 1] @ (A[t] @ x + B[t] @ u - f)

            new_carry = next_x
            new_output = (next_x, u)

            return new_carry, new_output

        (X, U) = jax.lax.scan(f, x0, jnp.arange(T), T)[1]

        return (jnp.concatenate([x0.reshape([1, n]), X]), U)

    f_0 = δ * v[0] - c[0]
    x0 = -F_inv[0] @ f_0

    X, U = rollout(K=K, k=k, x0=x0, A=A, B=B, c=c, F_inv=F_inv, v=v)

    Y = jax.vmap(lambda i: V[i] @ X[i] + v[i])(jnp.arange(N + 1))

    return X, U, Y
