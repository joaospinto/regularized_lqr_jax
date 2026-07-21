import jax

from functools import partial

from jax import numpy as jnp
from jax import scipy as jsp

from regularized_lqr_jax.types import (
    FactorizationInputs,
    SolveInputs,
    SolveOutputs,
)


@jax.jit
def symmetrize(x):
    return 0.5 * (x + jnp.swapaxes(x, -2, -1))


@jax.jit
def form_delta(Δ_L):
    """Compute Δ from a caller-supplied factor Δ_L with Δ = Δ_L Δ_L.T."""
    return Δ_L @ Δ_L.T


@jax.jit
def factor_symmetric_F(Δ_L, P):
    """Factor S = I + Δ_L.T P Δ_L."""
    S = symmetrize(jnp.eye(P.shape[-1], dtype=P.dtype) + Δ_L.T @ P @ Δ_L)
    S_cho = jsp.linalg.cho_factor(S, lower=True)[0]
    return S_cho


@jax.jit
def stable_F_solve(S_cho, Δ_L, P, b):
    """
    Solves F x = b through S = I + Δ_L.T P Δ_L, where Δ = Δ_L Δ_L.T.
    Uses F^{-1} = I - Δ_L S^{-1} Δ_L.T P, which permits singular Δ_L.
    """

    def symmetric_factor_solve(rhs):
        return rhs - Δ_L @ jsp.linalg.cho_solve((S_cho, True), Δ_L.T @ (P @ rhs))

    x = symmetric_factor_solve(b)
    residual = b - (jnp.eye(P.shape[-1], dtype=P.dtype) + form_delta(Δ_L) @ P) @ x
    return x + symmetric_factor_solve(residual)


@jax.jit
def stable_compute_W(S_cho, Δ_L, P):
    """
    Computes W = P (I + Δ P)^-1 from S = I + Δ_L.T P Δ_L.
    """
    F_inv = stable_F_solve(S_cho, Δ_L, P, jnp.eye(P.shape[-1], dtype=P.dtype))
    W = P @ F_inv
    return symmetrize(W)


@jax.jit
def factor_value_node(Δ_L, P):
    """Factor ``I + ΔP`` and form the regularized value curvature ``W``."""
    S_cho = factor_symmetric_F(Δ_L, P)
    W = stable_compute_W(S_cho, Δ_L, P)
    return W, S_cho


@jax.jit
def factor_feedback(A, B, M, R, W_child):
    """Factor one control block given its child's regularized value."""
    BtW = B.T @ W_child
    H = BtW @ A + M.T
    G = symmetrize(R + BtW @ B)
    G_cho = jsp.linalg.cho_factor(G, lower=False)[0]
    K = -jsp.linalg.cho_solve((G_cho, False), H)
    return K, G_cho, H


@jax.jit
def solve_feedforward_from_h(G_cho, h):
    """Solve a control feedforward block from its assembled RHS."""
    return -jsp.linalg.cho_solve((G_cho, False), h)


@jax.jit
def solve_feedforward(B, r, W_child, G_cho, f_child, p_child):
    """Assemble and solve one RHS-dependent control feedforward block."""
    g_child = p_child - W_child @ f_child
    h = r + B.T @ g_child
    return solve_feedforward_from_h(G_cho, h)


@jax.jit
def make_edge_value(A, B, M, R, Δ_child):
    """Eliminate one control to form a conditional edge value ``(A,C,P)``."""
    R_cho = jsp.linalg.cho_factor(symmetrize(R), lower=True)[0]
    RiMt = jsp.linalg.cho_solve((R_cho, True), M.T)
    RiBt = jsp.linalg.cho_solve((R_cho, True), B.T)
    return (
        A - B @ RiMt,
        symmetrize(Δ_child + B @ RiBt),
        symmetrize(-M @ RiMt),
    )


@jax.jit
def terminalize_value(path, terminal):
    """Fold a quadratic terminal value into a conditional edge/path value."""
    A, C, P = path
    system = jnp.eye(A.shape[-1], dtype=A.dtype) + C @ terminal
    solved_A = jsp.linalg.lu_solve(jsp.linalg.lu_factor(system), A)
    return symmetrize(P + A.T @ terminal @ solved_A)


@jax.jit
def compose_value_functions(left, right):
    """Compose adjacent conditional quadratic values in path order."""
    A_left, C_left, P_left = left
    A_right, C_right, P_right = right
    system = jnp.eye(A_left.shape[-1], dtype=A_left.dtype) + C_left @ P_right
    system_lu = jsp.linalg.lu_factor(system)
    solved_A = jsp.linalg.lu_solve(system_lu, A_left)
    solved_C = jsp.linalg.lu_solve(system_lu, C_left)
    return (
        A_right @ solved_A,
        symmetrize(A_right @ solved_C @ A_right.T + C_right),
        symmetrize(P_left + A_left.T @ P_right @ solved_A),
    )


@jax.jit
def project_psd_cone(Q, delta=0.0):
    """Projects Q into the positive semi-definite matrix cone."""
    S, V = jnp.linalg.eigh(symmetrize(Q))
    S = jnp.maximum(S, delta)
    Q_plus = jnp.matmul(V, jnp.matmul(jnp.diag(S), V.T))
    return symmetrize(Q_plus)


@jax.jit
def regularize(Q, R, M, psd_delta):
    """Regularizes the Q and R matrices.

    Args:
      Q:             [T+1, n, n]      numpy array.
      R:             [T, m, m]        numpy array.
      M:             [T, n, m]        numpy array.
      psd_delta:     the minimum eigenvalue post PSD cone projection.

    Returns:
      Q:             [T+1, n, n]      numpy array.
      R:             [T, m, m]        numpy array.
    """
    T, n, m = M.shape
    psd = jax.vmap(partial(project_psd_cone, delta=psd_delta))

    # This is done to ensure that the R are positive definite.
    R = psd(R)

    # This is done to ensure that the Q - M R^(-1) M^T are positive semi-definite.
    # M R^-1 M^T = (L^-1 M^T)^T (L^-1 M^T) where R = L L^T.
    L = jax.vmap(lambda R: jnp.linalg.cholesky(R))(R)
    LinvMT = jax.vmap(lambda L, M: jsp.linalg.solve_triangular(L, M.T, lower=True))(
        L, M
    )
    MRinvMT = jax.vmap(lambda LinvMT: LinvMT.T @ LinvMT)(LinvMT)

    QMRinvMT = jax.vmap(lambda Q, MRinvMT: Q - MRinvMT)(Q[:-1], MRinvMT)
    QMRinvMT = psd(QMRinvMT)
    Q_T = Q[T].reshape([1, n, n])
    Q_T = psd(Q_T)
    Q = jnp.concatenate([QMRinvMT + MRinvMT, Q_T])

    return Q, R


@jax.jit
def compute_residual(
    factorization_inputs: FactorizationInputs,
    solve_inputs: SolveInputs,
    solve_outputs: SolveOutputs,
):
    """Compute the residual of the regularized LQR system."""
    A = factorization_inputs.A
    B = factorization_inputs.B
    Q = factorization_inputs.Q
    M = factorization_inputs.M
    R = factorization_inputs.R
    Δ_L = factorization_inputs.Δ_L
    Δ = jax.vmap(form_delta)(Δ_L)

    q = solve_inputs.q
    r = solve_inputs.r
    c = solve_inputs.c

    X = solve_outputs.X
    U = solve_outputs.U
    Y = solve_outputs.Y

    T = A.shape[0]

    return jnp.concatenate(
        [
            jax.vmap(
                lambda Q, X, M, U, Y, A, Y_next, q: Q @ X + M @ U - Y + A.T @ Y_next + q
            )(Q[:-1], X[:-1], M, U, Y[:-1], A, Y[1:], q[:-1]).flatten(),
            jax.vmap(
                lambda M, X, R, U, B, Y_next, r: M.T @ X + R @ U + B.T @ Y_next + r
            )(M, X[:-1], R, U, B, Y[1:], r).flatten(),
            (Q[T] @ X[T] - Y[T] + q[T]),
            (-X[0] - Δ[0] @ Y[0] + c[0]),
            jax.vmap(
                lambda A, X, B, U, X_next, c_next, Δ_next, Y_next: (
                    A @ X + B @ U - X_next + c_next - Δ_next @ Y_next
                )
            )(A, X[:-1], B, U, X[1:], c[1:], Δ[1:], Y[1:]).flatten(),
        ]
    )
