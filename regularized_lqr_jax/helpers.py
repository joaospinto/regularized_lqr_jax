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
    return 0.5 * (x + x.T)


@jax.jit
def stable_F_solve(S_cho, L, b):
    """
    Solves F x = b,
    where F = I + Δ P, Δ = L L.T,
    and S_cho is the Cholesky factor of (I + L.T P L),
    using the similarity transformation
    (I + L L.T P) = L (I + L.T P L) L^-1,
    resulting in
    x = L (I + L.T P L)^-1 L^-1 b.
    """
    # y = L^-1 b
    y = jsp.linalg.solve_triangular(L, b, lower=True)
    # z = S^-1 y
    z = jsp.linalg.cho_solve((S_cho, False), y)
    # x = L z
    return L @ z


@jax.jit
def stable_compute_W(S_cho, L, PL):
    """
    Computes W = P (I + L L.T P)^-1 = P L (I + L.T P L)^-1 L^-1
    where S_cho is the Cholesky factor of (I + L.T P L) and PL = P @ L.
    """
    # V = (P L) S^-1
    # Solve S V.T = (P L).T
    VT = jsp.linalg.cho_solve((S_cho, False), PL.T)
    # W = V L^-1 => W L = V => L.T W.T = V.T => L.T W = V.T
    W = jsp.linalg.solve_triangular(L, VT, lower=True, trans="T").T
    return symmetrize(W)


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
    Δ = factorization_inputs.Δ

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
                lambda A, X, B, U, X_next, c_next, Δ_next, Y_next: A @ X
                + B @ U
                - X_next
                + c_next
                - Δ_next @ Y_next
            )(A, X[:-1], B, U, X[1:], c[1:], Δ[1:], Y[1:]).flatten(),
        ]
    )
