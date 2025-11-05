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
    Δ: jnp.ndarray,
):
    """
    Solves the following regularized LQR problem:
        [P   C.T] [x] = -[s],
        [C   -Δ ] [y]    [c]
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
    and Δ = diag(δ_0, ..., δ_N), with δ_0, ..., δ_N > 0.

    Shapes (where n, m are the state and control dimensions):
        A: [N, n, n],
        B: [N, n, m],
        Q: [N+1, n, n],
        M: [N, n, m],
        R: [N, m, m],
        q: [N+1, n],
        r: [N, m],
        c: [N+1, n],
        Δ: [N+1],

    Returns:
        X: [N+1, n],
        U: [N, m],
        Y: [N+1, n],
        V: [N+1, n, n],
        v: [N+1, n],
        K: [N, m, n],
        k: [N, m],
        respectively the states, controls, co-states, quadratic values (V, v),
        optimal control law (K, k).
    """
    N, n, m = B.shape

    def reg_lqr_step(V, v, F_inv, Q, M, R, A, B, q, r, c, δ, δ_next):
        """Performs a single step of the regularized LQR backward pass."""
        I_n = jnp.eye(n)
        W = F_inv @ V
        G = symmetrize(B.T @ W @ B + R)
        g = v + W @ (c - δ_next * v)
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
            V, v, F_inv, Q[i], M[i], R[i], A[i], B[i], q[i], r[i], c[i + 1], Δ[i], Δ[i + 1]
        )

        new_carry = (V, v, F_inv)
        new_output = (K, k, V, v, F_inv)

        return new_carry, new_output

    V_N = Q[N]
    v_N = q[N]
    I_n = jnp.eye(n)
    F_inv_N = jnp.linalg.inv(I_n + Δ[N] * V_N)

    K, k, V, v, F_inv = jax.lax.scan(
        f, (V_N, v_N, F_inv_N), jnp.arange(N), N, reverse=True
    )[1]

    V = jnp.concatenate([V, V_N.reshape([1, n, n])])
    v = jnp.concatenate([v, v_N.reshape([1, n])])
    F_inv = jnp.concatenate([F_inv, F_inv_N.reshape([1, n, n])])

    def rollout(K, k, x0, A, B, c, F_inv, v, Δ):
        """
        Performs the regularized LQR forward pass.
        """

        T, n, m = B.shape

        def f(carry, elem):
            t = elem

            x = carry
            u = K[t] @ x + k[t]
            f = Δ[t + 1] * v[t + 1] - c[t + 1]
            next_x = F_inv[t + 1] @ (A[t] @ x + B[t] @ u - f)

            new_carry = next_x
            new_output = (next_x, u)

            return new_carry, new_output

        (X, U) = jax.lax.scan(f, x0, jnp.arange(T), T)[1]

        return (jnp.concatenate([x0.reshape([1, n]), X]), U)

    f_0 = Δ[0] * v[0] - c[0]
    x0 = -F_inv[0] @ f_0

    X, U = rollout(K=K, k=k, x0=x0, A=A, B=B, c=c, F_inv=F_inv, v=v, Δ=Δ)

    Y = jax.vmap(lambda i: V[i] @ X[i] + v[i])(jnp.arange(N + 1))

    return X, U, Y, V, v, K, k


@jax.jit
def solve_parallel(
    A: jnp.ndarray,
    B: jnp.ndarray,
    Q: jnp.ndarray,
    M: jnp.ndarray,
    R: jnp.ndarray,
    q: jnp.ndarray,
    r: jnp.ndarray,
    c: jnp.ndarray,
    Δ: jnp.ndarray,
):
    """
    This is a O(log(N) * (log(n) + log(n) parallel time complexity implementation of `solve`.
    """
    T = Q.shape[0] - 1
    n = Q.shape[1]

    def fn(next, prev):
        def decompose(elem):
            return (
                elem[:n],
                elem[n],
                elem[n + 1 : 2 * n + 1],
                elem[2 * n + 1],
                elem[-n:],
            )

        A_l, c_l, C_l, p_l, P_l = decompose(prev)
        A_r, c_r, C_r, p_r, P_r = decompose(next)

        ArIClPr_inv = A_r @ jnp.linalg.inv(jnp.eye(n) + C_l @ P_r)
        AlTIPrCl_inv = A_l.T @ jnp.linalg.inv(jnp.eye(n) + P_r @ C_l)

        A_new = ArIClPr_inv @ A_l
        c_new = ArIClPr_inv @ (c_l - C_l @ p_r) + c_r
        C_new = ArIClPr_inv @ C_l @ A_r.T + C_r
        p_new = AlTIPrCl_inv @ (p_r + P_r @ c_l) + p_l
        P_new = AlTIPrCl_inv @ P_r @ A_l + P_l

        return jnp.concatenate(
            [
                A_new,
                c_new.reshape(1, n),
                C_new,
                p_new.reshape(1, n),
                P_new,
            ]
        )

    def chol_inv(t):
        m = R[t].shape[0]
        return solve_symmetric_positive_definite_system(R[t], jnp.eye(m))
    Rinv = jax.vmap(chol_inv)(jnp.arange(T))
    BRinv = jax.vmap(lambda t: B[t] @ Rinv[t])(jnp.arange(T))
    MRinv = jax.vmap(lambda t: M[t] @ Rinv[t])(jnp.arange(T))

    elems = jnp.concatenate(
        [
            # The A matrices.
            jnp.concatenate(
                [
                    A - jax.vmap(lambda t: BRinv[t] @ M[t].T)(jnp.arange(T)),
                    jnp.zeros([1, n, n]),
                ]
            ),
            # The c vectors (b, in the notation of https://ieeexplore.ieee.org/document/9697418).
            jnp.concatenate(
                [
                    (jax.vmap(lambda t: c[t + 1] - BRinv[t] @ r[t])(jnp.arange(T))).reshape(
                        [T, 1, n]
                    ),
                    jnp.zeros([1, 1, n]),
                ]
            ),
            # The C matrices.
            jnp.concatenate(
                [
                    jax.vmap(lambda t: Δ[t + 1] * jnp.eye(n) + BRinv[t] @ B[t].T)(jnp.arange(T)),
                    jnp.zeros([1, n, n]),
                ]
            ),
            # The p vectors (-eta, in the notation of https://ieeexplore.ieee.org/document/9697418).
            q.reshape([T + 1, 1, n])
            - jnp.concatenate(
                [
                    jax.vmap(lambda t: MRinv[t] @ r[t])(jnp.arange(T)).reshape([T, 1, n]),
                    jnp.zeros([1, 1, n]),
                ]
            ),
            # The P matrices (J, in the notation of https://ieeexplore.ieee.org/document/9697418).
            Q
            - jnp.concatenate(
                [
                    jax.vmap(lambda t: MRinv[t] @ M[t].T)(jnp.arange(T)),
                    jnp.zeros([1, n, n]),
                ]
            ),
        ],
        axis=1,
    )

    result = jax.lax.associative_scan(lambda r, l: jax.vmap(fn)(r, l), elems, reverse=True)

    P = result[:, -n:, :]
    p = result[:, 2 * n + 1, :]

    F = jax.vmap(lambda i: jnp.eye(n) + Δ[i] * P[i])(jnp.arange(T + 1))
    f = jax.vmap(lambda i: Δ[i] * p[i] - c[i])(jnp.arange(T + 1))

    def getKs(t):
        symmetrize = lambda x: 0.5 * (x + x.T)

        W = solve_symmetric_positive_definite_system(F[t + 1], P[t + 1])

        BtW = B[t].T @ W
        BtWA = BtW @ A[t]

        g = p[t + 1] + W @ (c[t + 1] - Δ[t + 1] * p[t + 1])

        H = BtWA + M[t].T
        h = r[t] + B[t].T @ g

        G = symmetrize(R[t] + BtW @ B[t])

        K_k = solve_symmetric_positive_definite_system(G, -jnp.hstack((H, h.reshape([-1, 1]))))
        K = K_k[:, :-1]
        k = K_k[:, -1]

        return K, k

    K, k = jax.vmap(getKs)(jnp.arange(T))

    def rollout_parallel(K, k, A, B, c, F, f, Δ):
        """Rolls-out time-varying linear policy u[t] = K[t] x[t] + k[t]."""
        T, _, n = K.shape

        def fn(prev, next):
            _F = prev[:-1]
            _f = prev[-1]
            _G = next[:-1]
            _g = next[-1]
            return jnp.concatenate([_G @ _F, (_g + _G @ _f).reshape([1, n])])

        # x_0 = -F_0^{-1} f_0
        x0 = solve_symmetric_positive_definite_system(F[0], -f[0])

        def get_elem(t):
            # x_{i+1} = F_{i+1}^{-1} (A_i x_i + B_i u_i - f_{i+1})
            # u_i = K_i x_i + k_i
            # x_{i+1} = F_{i+1}^{-1} ((A_i + B_i K_i) x_i + B_i k_i - f_{i+1})
            F_inv_ApBK = solve_symmetric_positive_definite_system(
                F[t + 1],
                A[t] + B[t] @ K[t],
            )
            F_inv_Bkmf = solve_symmetric_positive_definite_system(
                F[t + 1],
                B[t] @ k[t] - f[t + 1],
            )
            return jnp.concatenate(
                [F_inv_ApBK, F_inv_Bkmf.reshape([1, n])]
            )
        elems = jax.vmap(get_elem)(jnp.arange(T))
        comp = jax.lax.associative_scan(lambda l, r: jax.vmap(fn)(l, r), elems)
        X = jnp.concatenate(
            [
                x0.reshape(1, n),
                jax.vmap(lambda t: comp[t, :-1, :] @ x0 + comp[t, -1, :])(jnp.arange(T)),
            ]
        )

        U = jax.vmap(lambda t: K[t] @ X[t] + k[t])(jnp.arange(T))

        return X, U

    X, U = rollout_parallel(K=K, k=k, A=A, B=B, c=c, F=F, f=f, Δ=Δ)

    Y = jax.vmap(lambda i: P[i] @ X[i] + p[i])(jnp.arange(T + 1))

    return X, U, Y, P, p, K, k
