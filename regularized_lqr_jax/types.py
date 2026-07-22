import jax

from dataclasses import dataclass


@jax.tree_util.register_dataclass
@dataclass
class FactorizationInputs:
    """
    LHS data for a chain or rooted tree regularized LQR problem.

    For a tree, node arrays use original node order and edge arrays use
    contraction plan's ``edge_children`` order. Shapes (where n, m are the
    state and control dimensions, V nodes, and E edges; for a chain E=N and
    V=N+1):
        A: [E, n, n],
        B: [E, n, m],
        Q: [V, n, n],
        M: [E, n, m],
        R: [E, m, m],
        Δ_L: [V, n, n],
    """

    A: jax.Array
    B: jax.Array
    Q: jax.Array
    M: jax.Array
    R: jax.Array
    Δ_L: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class FactorizationOutputs:
    """
    Reusable compositional factorization for a chain or rooted tree.

    Shapes (V nodes and E edges):
        P: [V, n, n],
        K: [E, m, n],
        W: [V, n, n],
        G_cho: [E, m, m],
        S_cho: [V, n, n],
        ApBK: [E, n, n] for compositional plans, otherwise None,
        F_inv_ApBK: [E, n, n] for compositional plans, otherwise None,
    """

    P: jax.Array
    K: jax.Array
    W: jax.Array
    G_cho: jax.Array
    S_cho: jax.Array
    ApBK: jax.Array | None
    F_inv_ApBK: jax.Array | None


# Compatibility import names. Sequential and parallel execution now use the
# same canonical factorization representation and numerical implementation.
SequentialFactorizationOutputs = FactorizationOutputs
ParallelFactorizationOutputs = FactorizationOutputs


@jax.tree_util.register_dataclass
@dataclass
class SolveInputs:
    """
    RHS data for a chain or rooted tree regularized LQR problem.

    For a tree, node arrays use original node order and edge arrays use the
    contraction plan's edge order. Shapes:
        q: [V, n],
        r: [E, m],
        c: [V, n],
    """

    q: jax.Array
    r: jax.Array
    c: jax.Array


@jax.tree_util.register_dataclass
@dataclass
class SolveOutputs:
    """
    Solution for a chain or rooted tree (V nodes and E edges).

    Shapes:
        X: [V, n],
        U: [E, m],
        Y: [V, n],
        p: [V, n],
        k: [E, m],
    """

    X: jax.Array
    U: jax.Array
    Y: jax.Array
    p: jax.Array
    k: jax.Array
