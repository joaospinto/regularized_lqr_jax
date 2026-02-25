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
    and Δ = block_diag(Δ_0, ..., Δ_N).

    Shapes (where n, m are the state and control dimensions):
        A: [N, n, n],
        B: [N, n, m],
        Q: [N+1, n, n],
        M: [N, n, m],
        R: [N, m, m],
        q: [N+1, n],
        r: [N, m],
        c: [N+1, n],
        Δ: [N+1, n, n],

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

    def reg_lqr_step(V, v, F_inv, Q, M, R, A, B, q, r, c, Δ, Δ_next):
        """Performs a single step of the regularized LQR backward pass."""
        I_n = jnp.eye(n)
        W = V @ F_inv
        G = symmetrize(B.T @ W @ B + R)
        g = v + W @ (c - Δ_next @ v)
        H = B.T @ W @ A + M.T
        h = r + B.T @ g
        K_k = solve_symmetric_positive_definite_system(
            G, -jnp.hstack((H, h.reshape([-1, 1])))
        )
        K = K_k[:, :-1]
        k = K_k[:, -1]
        V = symmetrize(A.T @ W @ A + Q + K.T @ H)
        v = q + A.T @ g + K.T @ h
        F_inv = jnp.linalg.inv(I_n + Δ @ V)
        return K, k, V, v, F_inv

    def reg_lqr_step_wrapper(carry, elem):
        """Wraps reg_lqr_step for jax.lax.scan."""
        V, v, F_inv = carry
        Q, M, R, A, B, q, r, c, Δ, Δ_next = elem

        K, k, V, v, F_inv = reg_lqr_step(
            V,
            v,
            F_inv,
            Q,
            M,
            R,
            A,
            B,
            q,
            r,
            c,
            Δ,
            Δ_next,
        )

        new_carry = (V, v, F_inv)
        new_output = (K, k, V, v, F_inv)

        return new_carry, new_output

    V_N = Q[N]
    v_N = q[N]
    I_n = jnp.eye(n)
    F_inv_N = jnp.linalg.inv(I_n + Δ[N] @ V_N)

    K, k, V, v, F_inv = jax.lax.scan(
        reg_lqr_step_wrapper,
        (V_N, v_N, F_inv_N),
        (Q[:-1], M, R, A, B, q[:-1], r, c[1:], Δ[:-1], Δ[1:]),
        N,
        reverse=True,
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
            K, k, Δ, v, c, F_inv, A, B = elem

            x = carry
            u = K @ x + k
            f = Δ @ v - c
            next_x = F_inv @ (A @ x + B @ u - f)

            new_carry = next_x
            new_output = (next_x, u)

            return new_carry, new_output

        (X, U) = jax.lax.scan(f, x0, (K, k, Δ[1:], v[1:], c[1:], F_inv[1:], A, B), T)[1]

        return (jnp.concatenate([x0.reshape([1, n]), X]), U)

    f_0 = Δ[0] @ v[0] - c[0]
    x0 = -F_inv[0] @ f_0

    X, U = rollout(K=K, k=k, x0=x0, A=A, B=B, c=c, F_inv=F_inv, v=v, Δ=Δ)

    Y = jax.vmap(lambda V, X, v: V @ X + v)(V, X, v)

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

    def value_combination(next, prev):
        A_l, c_l, C_l, p_l, P_l = prev
        A_r, c_r, C_r, p_r, P_r = next

        ArIClPr_inv = A_r @ jnp.linalg.inv(jnp.eye(n) + C_l @ P_r)
        AlTIPrCl_inv = A_l.T @ jnp.linalg.inv(jnp.eye(n) + P_r @ C_l)

        A_new = ArIClPr_inv @ A_l
        c_new = ArIClPr_inv @ (c_l - C_l @ p_r) + c_r
        C_new = ArIClPr_inv @ C_l @ A_r.T + C_r
        p_new = AlTIPrCl_inv @ (p_r + P_r @ c_l) + p_l
        P_new = AlTIPrCl_inv @ P_r @ A_l + P_l

        return (A_new, c_new, C_new, p_new, P_new)

    def chol_inv(R):
        m = R.shape[0]
        return solve_symmetric_positive_definite_system(R, jnp.eye(m))

    Rinv = jax.vmap(chol_inv)(R)
    BRinv = jax.vmap(lambda B, Rinv: B @ Rinv)(B, Rinv)
    MRinv = jax.vmap(lambda M, Rinv: M @ Rinv)(M, Rinv)

    # The A matrices.
    A_mod = jnp.concatenate(
        [
            A - jax.vmap(lambda BRinv, M: BRinv @ M.T)(BRinv, M),
            jnp.zeros([1, n, n]),
        ]
    )

    # The c vectors (b, in the notation of https://ieeexplore.ieee.org/document/9697418).
    c_mod = jnp.concatenate(
        [
            (jax.vmap(lambda c, BRinv, r: c - BRinv @ r)(c[1:], BRinv, r)).reshape(
                [T, n]
            ),
            jnp.zeros([1, n]),
        ]
    )

    # The C matrices.
    C_mod = jnp.concatenate(
        [
            jax.vmap(lambda Δ, BRinv, B: Δ + BRinv @ B.T)(Δ[1:], BRinv, B),
            jnp.zeros([1, n, n]),
        ]
    )

    # The p vectors (-eta, in the notation of https://ieeexplore.ieee.org/document/9697418).
    q_mod = q.reshape([T + 1, n]) - jnp.concatenate(
        [
            jax.vmap(lambda MRinv, r: MRinv @ r)(MRinv, r).reshape([T, n]),
            jnp.zeros([1, n]),
        ]
    )

    # The P matrices (J, in the notation of https://ieeexplore.ieee.org/document/9697418).
    Q_mod = Q - jnp.concatenate(
        [
            jax.vmap(lambda MRinv, M: MRinv @ M.T)(MRinv, M),
            jnp.zeros([1, n, n]),
        ]
    )

    _, _, _, p, P = jax.lax.associative_scan(
        jax.vmap(value_combination),
        (A_mod, c_mod, C_mod, q_mod, Q_mod),
        reverse=True,
    )

    F = jax.vmap(lambda Δ, P: jnp.eye(n) + Δ @ P)(Δ, P)
    f = jax.vmap(lambda Δ, p, c: Δ @ p - c)(Δ, p, c)

    def getKs(F, f, P, p, A, B, c, Δ, M, R, r):
        F_inv = jnp.linalg.inv(F)

        W = symmetrize(P @ F_inv)

        BtW = B.T @ W
        BtWA = BtW @ A

        g = p - W @ f

        H = BtWA + M.T
        h = r + B.T @ g

        G = symmetrize(R + BtW @ B)

        K_k = solve_symmetric_positive_definite_system(
            G, -jnp.hstack((H, h.reshape([-1, 1])))
        )
        K = K_k[:, :-1]
        k = K_k[:, -1]

        return K, k

    K, k = jax.vmap(getKs)(F[1:], f[1:], P[1:], p[1:], A, B, c[1:], Δ[1:], M, R, r)

    def rollout_parallel(K, k, A, B, c, F, f):
        """Rolls-out time-varying linear policy u[t] = K[t] x[t] + k[t]."""
        T, _, n = K.shape

        def affine_fn_combiner(prev, next):
            _F, _f = prev
            _G, _g = next
            return (_G @ _F, _g + _G @ _f)

        # x_0 = -F_0^{-1} f_0
        x0 = jnp.linalg.solve(F[0], -f[0])

        def get_forward_dynamics(F, f, K, k, A, B):
            # x_{i+1} = F_{i+1}^{-1} (A_i x_i + B_i u_i - f_{i+1})
            # u_i = K_i x_i + k_i
            # x_{i+1} = F_{i+1}^{-1} ((A_i + B_i K_i) x_i + B_i k_i - f_{i+1})
            F_inv_ApBK = jnp.linalg.solve(
                F,
                A + B @ K,
            )
            F_inv_Bkmf = jnp.linalg.solve(
                F,
                B @ k - f,
            )
            return (F_inv_ApBK, F_inv_Bkmf)

        forward_dynamics = jax.vmap(get_forward_dynamics)(F[1:], f[1:], K, k, A, B)
        composed_dynamics = jax.lax.associative_scan(
            jax.vmap(affine_fn_combiner), forward_dynamics
        )
        X = jnp.concatenate(
            [
                x0.reshape(1, n),
                jax.vmap(lambda A, b: A @ x0 + b)(
                    composed_dynamics[0],
                    composed_dynamics[1],
                ),
            ]
        )

        U = jax.vmap(lambda K, X, k: K @ X + k)(K, X[:-1], k)

        return X, U

    X, U = rollout_parallel(K=K, k=k, A=A, B=B, c=c, F=F, f=f)

    Y = jax.vmap(lambda P, X, p: P @ X + p)(P, X, p)

    return X, U, Y, P, p, K, k
