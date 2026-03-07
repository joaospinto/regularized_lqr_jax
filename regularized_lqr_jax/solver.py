import jax
from jax import numpy as jnp


from regularized_lqr_jax.helpers import (
    symmetrize,
    stable_F_solve,
    stable_compute_W,
)

from regularized_lqr_jax.types import (
    FactorizationInputs,
    SequentialFactorizationOutputs,
    ParallelFactorizationOutputs,
    SolveInputs,
    SolveOutputs,
)


@jax.jit
def factor(inputs: FactorizationInputs) -> SequentialFactorizationOutputs:
    """
    Factors the block-matrix
        [P   C.T],
        [C   -Δ ]
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
    and Δ = block_diag(Δ_0, ..., Δ_N).
    """
    A = inputs.A
    B = inputs.B
    Q = inputs.Q
    M = inputs.M
    R = inputs.R
    Δ = inputs.Δ

    N, n, m = B.shape

    L = jax.vmap(lambda Δ: jax.scipy.linalg.cho_factor(Δ, lower=True)[0])(Δ)

    def reg_lqr_step(carry, elem):
        """Performs a single partial step of the regularized LQR backward pass."""
        W_next = carry
        Q, M, R, A, B, L = elem

        I_n = jnp.eye(n)
        G = symmetrize(B.T @ W_next @ B + R)
        G_cho = jax.scipy.linalg.cho_factor(G, lower=False)[0]
        H = B.T @ W_next @ A + M.T
        K = -jax.scipy.linalg.cho_solve((G_cho, False), H)
        P = symmetrize(A.T @ W_next @ A + Q + K.T @ H)

        PL = P @ L
        S = I_n + L.T @ PL
        S_cho = jax.scipy.linalg.cho_factor(S, lower=False)[0]
        W = stable_compute_W(S_cho, L, PL)

        new_carry = W
        new_output = (W, G_cho, K, P, S_cho)

        return new_carry, new_output

    V_N = Q[N]
    I_n = jnp.eye(n)
    L_N = L[N]
    PL_N = V_N @ L_N
    S_N = I_n + L_N.T @ PL_N
    S_cho_N = jax.scipy.linalg.cho_factor(S_N, lower=False)[0]
    W_N = stable_compute_W(S_cho_N, L_N, PL_N)

    W, G_cho, K, P, S_cho = jax.lax.scan(
        reg_lqr_step,
        W_N,
        (Q[:-1], M, R, A, B, L[:-1]),
        N,
        reverse=True,
    )[1]

    W = jnp.concatenate([W, W_N.reshape([1, n, n])])
    P = jnp.concatenate([P, V_N.reshape([1, n, n])])
    S_cho = jnp.concatenate([S_cho, S_cho_N.reshape([1, n, n])])

    return SequentialFactorizationOutputs(P, K, W, G_cho, L, S_cho)


@jax.jit
def solve(
    factorization_inputs: FactorizationInputs,
    factorization_outputs: SequentialFactorizationOutputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
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
    """
    A = factorization_inputs.A
    B = factorization_inputs.B
    Δ = factorization_inputs.Δ

    P = factorization_outputs.P
    K = factorization_outputs.K
    W = factorization_outputs.W
    G_cho = factorization_outputs.G_cho
    L = factorization_outputs.L
    S_cho = factorization_outputs.S_cho

    q = solve_inputs.q
    r = solve_inputs.r
    c = solve_inputs.c

    N, n, m = B.shape

    def reg_lqr_step(carry, elem):
        """Performs a single partial step of the regularized LQR backward pass."""
        p_next = carry
        A, B, q, r, c_next, Δ_next, W_next, G_cho, K = elem

        f_next = Δ_next @ p_next - c_next
        g_next = p_next - W_next @ f_next
        h = r + B.T @ g_next
        k = -jax.scipy.linalg.cho_solve((G_cho, False), h)
        p = q + A.T @ g_next + K.T @ h

        new_carry = p
        new_output = (f_next, k, p)

        return new_carry, new_output

    p_N = q[N]
    f, k, p = jax.lax.scan(
        reg_lqr_step,
        p_N,
        (A, B, q[:-1], r, c[1:], Δ[1:], W[1:], G_cho, K),
        N,
        reverse=True,
    )[1]
    p = jnp.concatenate([p, p_N.reshape([1, n])])
    f_0 = Δ[0] @ p[0] - c[0]
    f = jnp.concatenate([f_0.reshape([1, n]), f])

    x0 = -stable_F_solve(S_cho[0], L[0], f_0)

    def forward_dynamics(carry, elem):
        K, k, Δ, p, c, f, S_cho, L, P, A, B = elem

        x = carry
        u = K @ x + k
        next_x = stable_F_solve(S_cho, L, A @ x + B @ u - f)

        new_carry = next_x
        new_output = (next_x, u)

        return new_carry, new_output

    X, U = jax.lax.scan(
        forward_dynamics,
        x0,
        (K, k, Δ[1:], p[1:], c[1:], f[1:], S_cho[1:], L[1:], P[1:], A, B),
        N,
    )[1]

    X = jnp.concatenate([x0.reshape([1, n]), X])

    Y = jax.vmap(lambda P, X, p: P @ X + p)(P, X, p)

    return SolveOutputs(X, U, Y, p, k)


@jax.jit
def factor_parallel(inputs: FactorizationInputs) -> ParallelFactorizationOutputs:
    """
    This is a O(log(N) * (log^2(n) + log^2(n)) parallel time complexity implementation of `factor`.
    """
    A = inputs.A
    B = inputs.B
    Q = inputs.Q
    M = inputs.M
    R = inputs.R
    Δ = inputs.Δ

    N, n, m = B.shape

    R_cho = jax.vmap(lambda R: jax.scipy.linalg.cho_factor(R, lower=True)[0])(R)
    BR_inv = jax.vmap(
        lambda R_cho, B: jax.scipy.linalg.cho_solve((R_cho, True), B.T).T
    )(R_cho, B)
    MR_inv = jax.vmap(
        lambda R_cho, M: jax.scipy.linalg.cho_solve((R_cho, True), M.T).T
    )(R_cho, M)

    # The A matrices.
    A_mod = jnp.concatenate(
        [
            A - jax.vmap(lambda BR_inv, M: BR_inv @ M.T)(BR_inv, M),
            jnp.zeros([1, n, n]),
        ]
    )

    # The C matrices.
    C_mod = jnp.concatenate(
        [
            jax.vmap(lambda Δ, BR_inv, B: Δ + BR_inv @ B.T)(Δ[1:], BR_inv, B),
            jnp.zeros([1, n, n]),
        ]
    )

    # The P matrices (J, in the notation of https://ieeexplore.ieee.org/document/9697418).
    Q_mod = Q - jnp.concatenate(
        [
            jax.vmap(lambda MR_inv, M: MR_inv @ M.T)(MR_inv, M),
            jnp.zeros([1, n, n]),
        ]
    )

    def value_combination(next, prev):
        A_l, C_l, P_l = prev
        A_r, C_r, P_r = next

        ArIClPr_inv = jnp.linalg.solve(jnp.eye(n) + P_r.T @ C_l.T, A_r.T).T
        AlTIPrCl_inv = jnp.linalg.solve(jnp.eye(n) + C_l.T @ P_r.T, A_l).T

        A_new = ArIClPr_inv @ A_l
        C_new = symmetrize(ArIClPr_inv @ C_l @ A_r.T + C_r)
        P_new = symmetrize(AlTIPrCl_inv @ P_r @ A_l + P_l)

        return (A_new, C_new, P_new)

    _, _, P = jax.lax.associative_scan(
        jax.vmap(value_combination),
        (A_mod, C_mod, Q_mod),
        reverse=True,
    )

    L = jax.vmap(lambda Δ: jax.scipy.linalg.cho_factor(Δ, lower=True)[0])(Δ)
    PL = jax.vmap(lambda P, L: P @ L)(P, L)
    S = jax.vmap(lambda L, PL: jnp.eye(n) + symmetrize(L.T @ PL))(L, PL)
    S_cho = jax.vmap(lambda S: jax.scipy.linalg.cho_factor(S, lower=False)[0])(S)
    W = jax.vmap(stable_compute_W)(S_cho, L, PL)

    def get_Ks_and_Gchos(W, P, A, B, Δ, M, R):
        BtW = B.T @ W
        BtWA = BtW @ A
        H = BtWA + M.T
        G = symmetrize(R + BtW @ B)
        G_cho = jax.scipy.linalg.cho_factor(G, lower=False)[0]
        K = -jax.scipy.linalg.cho_solve((G_cho, False), H)
        return K, G_cho

    K, G_cho = jax.vmap(get_Ks_and_Gchos)(W[1:], P[1:], A, B, Δ[1:], M, R)

    # x_{i+1} = F_{i+1}^{-1} (A_i x_i + B_i u_i - f_{i+1})
    # u_i = K_i x_i + k_i
    # x_{i+1} = F_{i+1}^{-1} ((A_i + B_i K_i) x_i + B_i k_i - f_{i+1})
    ApBK = jax.vmap(lambda K, A, B: (A + B @ K))(K, A, B)
    F_inv_ApBK = jax.vmap(stable_F_solve)(S_cho[1:], L[1:], ApBK)

    return ParallelFactorizationOutputs(P, K, W, G_cho, L, S_cho, ApBK, F_inv_ApBK)


@jax.jit
def solve_parallel(
    factorization_inputs: FactorizationInputs,
    factorization_outputs: ParallelFactorizationOutputs,
    solve_inputs: SolveInputs,
) -> SolveOutputs:
    """
    This is a O(log(N) * (log(n) + log(n)) parallel time complexity implementation of `solve`.
    """
    B = factorization_inputs.B
    Δ = factorization_inputs.Δ

    P = factorization_outputs.P
    K = factorization_outputs.K
    W = factorization_outputs.W
    G_cho = factorization_outputs.G_cho
    L = factorization_outputs.L
    S_cho = factorization_outputs.S_cho
    ApBK = factorization_outputs.ApBK
    F_inv_ApBK = factorization_outputs.F_inv_ApBK

    q = solve_inputs.q
    r = solve_inputs.r
    c = solve_inputs.c

    N, n, m = B.shape

    # Recover p via the affine recurrence
    # p_i = Z_i p_{i+1} + z_i, where
    # Z_i = (F_{i+1}^{-1} (A_i + B_i K_i))^T and
    # z_i = q_i + K_i^T r_i + (A_i + B_i K_i)^T (W_{i+1} c_{i+1}).
    Z = F_inv_ApBK.mT
    z = jax.vmap(lambda q, r, c, ApBK, W, K: q + K.T @ r + ApBK.T @ (W @ c))(
        q[:-1], r, c[1:], ApBK, W[1:], K
    )

    def affine_fn_composer(_Ff, _Gg):
        _F, _f = _Ff
        _G, _g = _Gg
        return _G @ _F, _G @ _f + _g

    Z_scan, z_scan = jax.lax.associative_scan(
        jax.vmap(affine_fn_composer), (Z, z), reverse=True
    )
    p_N = q[N]
    p = jax.vmap(lambda Z, z: Z @ p_N + z)(Z_scan, z_scan)
    p = jnp.concatenate([p, p_N.reshape([1, n])])

    f = jax.vmap(lambda Δ, p, c: Δ @ p - c)(Δ, p, c)

    def get_k(B, r, W, G_cho, f, p):
        g = p - W @ f
        h = r + B.T @ g
        k = -jax.scipy.linalg.cho_solve((G_cho, False), h)
        return k

    k = jax.vmap(get_k)(B, r, W[1:], G_cho, f[1:], p[1:])

    # x_0 = -F_0^{-1} f_0
    x0 = -stable_F_solve(S_cho[0], L[0], f[0])

    # x_{i+1} = F_{i+1}^{-1} (A_i x_i + B_i u_i - f_{i+1})
    # u_i = K_i x_i + k_i
    # x_{i+1} = F_{i+1}^{-1} ((A_i + B_i K_i) x_i + B_i k_i - f_{i+1})
    F_inv_Bkmf = jax.vmap(
        lambda S_cho, L, B, k, f: stable_F_solve(S_cho, L, B @ k - f)
    )(S_cho[1:], L[1:], B, k, f[1:])

    composed_dynamics = jax.lax.associative_scan(
        jax.vmap(affine_fn_composer), (F_inv_ApBK, F_inv_Bkmf)
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

    Y = jax.vmap(lambda P, X, p: P @ X + p)(P, X, p)

    return SolveOutputs(X, U, Y, p, k)
