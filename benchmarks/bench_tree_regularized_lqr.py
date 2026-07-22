#!/usr/bin/env python3
"""Benchmark setup, factor, solve, and combined tree-LQR phases."""

from __future__ import annotations

import argparse
import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax_bidirectional_tree_rake_compress import plan_statistics

from regularized_lqr_jax.tree_solver import (
    factor_and_solve_tree,
    factor_tree,
    make_tree_lqr_plan,
    solve_tree,
)
from regularized_lqr_jax.types import FactorizationInputs, SolveInputs


def parents_for(topology: str, nodes: int) -> np.ndarray:
    parents = np.full(nodes, -1, dtype=np.int32)
    if topology == "chain":
        parents[1:] = np.arange(nodes - 1)
    elif topology == "balanced":
        parents[1:] = (np.arange(1, nodes) - 1) // 2
    elif topology == "star":
        parents[1:] = 0
    elif topology == "comb":
        for node in range(1, nodes):
            parents[node] = 0 if node == 1 else node - (1 if node % 2 == 0 else 2)
    else:
        raise ValueError(f"unknown topology: {topology}")
    return parents


def make_problem(plan, seed: int, n: int, m: int):
    key = jax.random.key(seed)
    A_key, B_key, q_key, r_key, c_key = jax.random.split(key, 5)
    V, E = plan.num_nodes, plan.num_edges
    lhs = FactorizationInputs(
        A=0.25 * jax.random.normal(A_key, (E, n, n)),
        B=0.35 * jax.random.normal(B_key, (E, n, m)),
        Q=jnp.tile((2.0 * jnp.eye(n))[None], (V, 1, 1)),
        M=jnp.zeros((E, n, m)),
        R=jnp.tile(jnp.eye(m)[None], (E, 1, 1)),
        Δ_L=jnp.tile((jnp.sqrt(0.02) * jnp.eye(n))[None], (V, 1, 1)),
    )
    rhs = SolveInputs(
        q=jax.random.normal(q_key, (V, n)),
        r=jax.random.normal(r_key, (E, m)),
        c=jax.random.normal(c_key, (V, n)),
    )
    return lhs, rhs


def median_ms(function, repeats: int) -> float:
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(function())
        samples.append(1e3 * (time.perf_counter() - start))
    return statistics.median(samples)


def pretty_table(headers, rows):
    text_rows = [[str(value) for value in row] for row in rows]
    widths = [
        max(len(headers[column]), *(len(row[column]) for row in text_rows))
        for column in range(len(headers))
    ]
    rule = "+-" + "-+-".join("-" * width for width in widths) + "-+"
    print(rule)
    print(
        "| "
        + " | ".join(heading.ljust(width) for heading, width in zip(headers, widths))
        + " |"
    )
    print(rule)
    for row in text_rows:
        print(
            "| "
            + " | ".join(value.rjust(width) for value, width in zip(row, widths))
            + " |"
        )
    print(rule)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--m", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument(
        "--sizes",
        default="8,16,32,64,128,256,512,1024,2048,4096,8192",
    )
    parser.add_argument("--topologies", default="chain,balanced,comb,star")
    parser.add_argument(
        "--execution", choices=("sequential", "parallel"), default="parallel"
    )
    args = parser.parse_args()
    rows = []
    parallel = args.execution == "parallel"

    for topology in args.topologies.split(","):
        for nodes in (int(value) for value in args.sizes.split(",")):
            parents = parents_for(topology, nodes)
            plan = make_tree_lqr_plan(parents, parallel=parallel)
            lhs, rhs = make_problem(plan, seed=nodes, n=args.n, m=args.m)

            factorization = jax.block_until_ready(factor_tree(plan, lhs))
            jax.block_until_ready(solve_tree(plan, lhs, factorization, rhs))
            jax.block_until_ready(factor_and_solve_tree(plan, lhs, rhs))

            setup_ms = median_ms(
                lambda: make_tree_lqr_plan(parents, parallel=parallel), args.repeats
            )
            factor_ms = median_ms(lambda: factor_tree(plan, lhs), args.repeats)
            solve_ms = median_ms(
                lambda: solve_tree(plan, lhs, factorization, rhs),
                args.repeats,
            )
            combined_ms = median_ms(
                lambda: factor_and_solve_tree(plan, lhs, rhs),
                args.repeats,
            )

            def full_run():
                dynamic_plan = make_tree_lqr_plan(parents, parallel=parallel)
                return factor_and_solve_tree(dynamic_plan, lhs, rhs)

            full_ms = median_ms(full_run, args.repeats)
            rows.append(
                [
                    topology,
                    nodes,
                    plan_statistics(plan).num_rounds,
                    f"{setup_ms:.3f}",
                    f"{factor_ms:.3f}",
                    f"{solve_ms:.3f}",
                    f"{combined_ms:.3f}",
                    f"{full_ms:.3f}",
                ]
            )

    print(
        f"backend={jax.default_backend()} execution={args.execution} "
        f"n={args.n} m={args.m}"
    )
    pretty_table(
        [
            "topology",
            "nodes",
            "rounds",
            "setup ms",
            "factor ms",
            "solve ms",
            "factor+solve ms",
            "setup+factor+solve ms",
        ],
        rows,
    )


if __name__ == "__main__":
    main()
