# Regularized LQR on a rooted tree

## Problem and data layout

Let each non-root node (j) have one incoming edge (e=(i,j)). The solver
handles

\[
\begin{aligned}
\min_{x,u}\quad
&\sum_i \left(\tfrac12 x_i^TQ_ix_i+q_i^Tx_i\right)
 +\sum_{e=(i,j)}\left(
   x_i^TM_eu_e+\tfrac12u_e^TR_eu_e+r_e^Tu_e
 \right),\\
\text{subject to}\quad
&c_\rho-x_\rho-\Delta_\rho y_\rho=0,\\
&A_ex_i+B_eu_e+c_j-x_j-\Delta_jy_j=0,
\qquad e=(i,j),
\end{aligned}
\]

where (\Delta_i=\Delta_{L,i}\Delta_{L,i}^T). Setting (\Delta_{L,i}=0)
gives exact dynamics. The implementation assumes uniform state and control
dimensions, positive-definite (R_e), and the usual convexity conditions on
the quadratic objective.

`make_tree_lqr_plan(parents, parallel=...)` is the only topology preprocessing.
It runs on the host once and selects the contraction executor. Node arrays
retain caller order; edge arrays follow `plan.edge_children`, with endpoints
exposed by `plan.edge_parents` and `plan.edge_children`. Neither factor nor
solve performs host traversal.

## Quadratic factorization

Eliminating (u_e) independently turns each edge into the conditional value

\[
V_e(x_i,x_j)=\max_y\left\{
\tfrac12x_i^T\bar P_ex_i+y^T(\bar A_ex_i-x_j)
-\tfrac12y^T\bar C_ey
\right\},
\]

with

\[
\bar A_e=A_e-B_eR_e^{-1}M_e^T,\qquad
\bar C_e=\Delta_j+B_eR_e^{-1}B_e^T,\qquad
\bar P_e=-M_eR_e^{-1}M_e^T.
\]

When compression is possible, a path summary is the fixed-size triple
((A,C,P)); an active node stores its accumulated subtree quadratic (P_i). The
contraction rules are:

- **Rake:** fold a completed leaf value into its incoming path and send the
  resulting quadratic to its parent.
- **Branch reduction:** add all messages targeting the same parent.
- **Compress:** add the middle node's accumulated quadratic to the right path,
  then compose the left and right conditional values.
- **Expansion:** once the right endpoint value is known, fold it into the saved
  right path to recover the eliminated middle value.

The terminal fold is

\[
\mathcal T((A,C,P),P_t)=P+A^TP_t(I+CP_t)^{-1}A.
\]

For adjacent left and right paths, let (S=I+C_lP_r). Their composition is

\[
\begin{aligned}
A&=A_rS^{-1}A_l,\\
C&=C_r+A_rS^{-1}C_lA_r^T,\\
P&=P_l+A_l^TP_rS^{-1}A_l.
\end{aligned}
\]

One contraction and reverse expansion produce every (P_i). The remaining
factorization work is node- or edge-local:

\[
\begin{aligned}
W_j &= P_j(I+\Delta_jP_j)^{-1},\\
G_e &= R_e+B_e^TW_jB_e,\\
K_e &=-G_e^{-1}(B_e^TW_jA_e+M_e^T),\\
T_e &=(I+\Delta_jP_j)^{-1}(A_e+B_eK_e).
\end{aligned}

For a rake-only plan, the same contraction executor applies the equivalent
Riccati rake directly to each original edge. This retains the stable symmetric
factorizations used by the sequential method; it is a schedule-specific local
algebra inside the same `factor_tree`/`solve_tree` implementation, not a
separate chain solver.
\]

The implementation factors (I+\Delta_{L,j}^TP_j\Delta_{L,j}) instead of
forming an unstable inverse. Those Cholesky factors, (G_e)'s factors, and
(T_e) are cached for every subsequent RHS.

## Upward affine solve

For a new RHS, the affine cost-to-go vectors obey

\[
p_i=q_i+\sum_{e=(i,j)}(Z_ep_j+z_e),
\]

where

\[
Z_e=T_e^T,\qquad
z_e=K_e^Tr_e+(A_e+B_eK_e)^TW_jc_j.
\]

This is a second rake--compress algebra. A path stores an affine map; raked
branches add; compression composes the maps while adding the middle node's
accumulated RHS. Reverse expansion recovers all (p_i). Then

\[
f_j=\Delta_jp_j-c_j,\qquad
g_j=p_j-W_jf_j,\qquad
k_e=-G_e^{-1}(r_e+B_e^Tg_j).
\]

## Downward state recovery

The root and edge states satisfy

\[
x_\rho=-(I+\Delta_\rho P_\rho)^{-1}f_\rho,\qquad
x_j=T_ex_i+b_e,
\]

with

\[
b_e=(I+\Delta_jP_j)^{-1}(B_ek_e-f_j).
\]

Contracting these affine maps creates a recovery tape; expansion broadcasts the
root state to every node in logarithmic depth. Finally,

\[
u_e=K_ex_i+k_e,\qquad y_i=P_ix_i+p_i.
\]

## Code reuse and the chain cases

Sequential chains, parallel chains, sequential trees, and parallel trees use
the same `factor_tree` and `solve_tree` implementation. The `factor`, `solve`,
`factor_parallel`, and `solve_parallel` names only construct an implicit
root-at-zero chain plan and delegate to those functions.

A sequential Riccati pass can be viewed algebraically as repeatedly raking the
terminal leaf of a chain. A literal rake-only parallel schedule would still
have (O(N)) depth, however, so the implementation keeps `lax.scan` for the
sequential chain. The associative-scan chain path and rake--compress tree path
both have (O(\log N)) depth. On a pure chain, a work-efficient associative
scan has roughly the same total composition count and lower scheduling
overhead, so it remains the preferred chain specialization.

For the tree API, a sequential chain uses a rake-only `lax.scan`; a
parallel chain uses the associative-scan contraction executor. Branching trees
use the unrolled executor with rake-only or rake--compress scheduling according
to the requested mode. The chain API is only a convenience facade for callers
whose arrays are already in canonical chain order.
