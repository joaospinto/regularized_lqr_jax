import jax

from dataclasses import dataclass


@jax.tree_util.register_dataclass
@dataclass
class FactorizationInputs:
    """
    Shapes (where n, m are the state and control dimensions):
        A: [N, n, n],
        B: [N, n, m],
        Q: [N+1, n, n],
        M: [N, n, m],
        R: [N, m, m],
        Δ: [N+1, n, n],
    """

    A: jax.Array
    B: jax.Array
    Q: jax.Array
    M: jax.Array
    R: jax.Array
    Δ: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class SequentialFactorizationOutputs:
    """
    Shapes (where n, m are the state and control dimensions):
        P: [N+1, n, n],
        K: [N, m, n],
        W: [N+1, n, n],
        G_inv: [N, m, m],
        L: [N+1, n, n],
        S_cho: [N+1, n, n],
    """

    P: jax.Array
    K: jax.Array
    W: jax.Array
    G_inv: jax.Array
    L: jax.Array
    S_cho: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class ParallelFactorizationOutputs:
    """
    Shapes (where n, m are the state and control dimensions):
        P: [T+1, n, n],
        K: [T, m, n],
        W: [T+1, n, n],
        G_inv: [N, m, m],
        L: [N+1, n, n],
        S_cho: [N+1, n, n],
        ApBK: [N, n, m],
        F_inv_ApBK: [N, n, m],
    """

    P: jax.Array
    K: jax.Array
    W: jax.Array
    G_inv: jax.Array
    L: jax.Array
    S_cho: jax.Array
    ApBK: jax.Array
    F_inv_ApBK: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class SolveInputs:
    """
    Shapes (where n, m are the state and control dimensions):
        q: [N+1, n],
        r: [N, m],
        c: [N+1, n],
    """

    q: jax.Array
    r: jax.Array
    c: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class SolveOutputs:
    """
    Shapes (where n, m are the state and control dimensions):
        X: [N+1, n],
        U: [N, m],
        Y: [N+1, n],
        p: [N+1, n],
        k: [N, m],
    """

    X: jax.Array
    U: jax.Array
    Y: jax.Array
    p: jax.Array
    k: jax.Array
