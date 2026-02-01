import jax

from functools import partial

from jax import numpy as jnp
from jax import scipy as jsp


@jax.jit
def _2x2_inv(M):
    # See https://en.wikipedia.org/wiki/Adjugate_matrix.
    a, b, c, d = M.flatten()
    det = a * d - b * c
    return (1.0 / det) * jnp.array([[d, -b], [-c, a]])


@jax.jit
def _solve_cholesky(A, b):
    f = jsp.linalg.cho_factor(A)
    return jsp.linalg.cho_solve(f, b)


@jax.jit
def solve_symmetric_positive_definite_system(A, b):
    n, _ = A.shape
    if n == 2:
        return _2x2_inv(A) @ b
    return _solve_cholesky(A, b)


@jax.jit
def symmetrize(x):
    return 0.5 * (x + x.T)


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
    Rinv = jax.vmap(lambda t: jnp.linalg.inv(R[t]))(jnp.arange(T))
    MRinvMT = jax.vmap(lambda t: M[t] @ Rinv[t] @ M[t].T)(jnp.arange(T))
    QMRinvMT = jax.vmap(lambda t: Q[t] - MRinvMT[t])(jnp.arange(T))
    QMRinvMT = psd(QMRinvMT)
    Q_T = Q[T].reshape([1, n, n])
    Q_T = psd(Q_T)
    Q = jnp.concatenate([QMRinvMT + MRinvMT, Q_T])

    return Q, R


@jax.jit
def compute_residual(
    A: jnp.ndarray,
    B: jnp.ndarray,
    Q: jnp.ndarray,
    M: jnp.ndarray,
    R: jnp.ndarray,
    q: jnp.ndarray,
    r: jnp.ndarray,
    c: jnp.ndarray,
    X: jnp.ndarray,
    U: jnp.ndarray,
    Y: jnp.ndarray,
    Δ: jnp.ndarray,
):
    T = A.shape[0]

    return jnp.concatenate(
        [
            jax.vmap(
                lambda i: Q[i] @ X[i] + M[i] @ U[i] - Y[i] + A[i].T @ Y[i + 1] + q[i]
            )(jnp.arange(T)).flatten(),
            jax.vmap(lambda i: M[i].T @ X[i] + R[i] @ U[i] + B[i].T @ Y[i + 1] + r[i])(
                jnp.arange(T)
            ).flatten(),
            (Q[T] @ X[T] - Y[T] + q[T]),
            (-X[0] - Δ[0] * Y[0] + c[0]),
            jax.vmap(
                lambda i: A[i] @ X[i] + B[i] @ U[i] - X[i + 1] + c[i + 1] - Δ[i + 1] * Y[i + 1]
            )(jnp.arange(T)).flatten(),
        ]
    )
