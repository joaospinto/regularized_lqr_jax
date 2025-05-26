# Regularized LQR

This library implements a regularized LQR solver in JAX.

## Introduction

A regularized LQR problem is a linear system of the form

$$
\begin{bmatrix}
P & C^T \\
C & -\delta I
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix} = -
\begin{bmatrix}
s \\
c
\end{bmatrix},
$$
where
$$
\begin{align*}
P &=
\begin{bmatrix}
P_0 & & \\
& \ddots & \\
& & P_N
\end{bmatrix}, \\
P_i &=
\begin{cases}
\begin{bmatrix}
Q_i & M_i \\
M_i^T & R_i
\end{bmatrix}, & \text{if } 0 \leq i \lt N, \\
Q_i, & \text{if } i = N,
\end{cases} \\
C &=
\begin{bmatrix}
-I  &     &     &     &     &     &    &        &         &         & \\
A_0 & B_0 &  -I &     &     &     &    &        &         &         & \\
    &     & A_1 & B_1 &  -I       &    &        &         &         & \\
    &     &     &     & A_2 & B_2 & -I &        &         &         & \\
    &     &     &     &           &    & \ddots &      -I &         & \\
    &     &     &     &           &    &        & A_{N-1} & B_{N-1} & -I
\end{bmatrix}, \\
s &=
\begin{bmatrix}
q_0 \\
r_0 \\
\vdots \\
q_{N-1} \\
r_{N-1} \\
q_N
\end{bmatrix}, \\
c &=
\begin{bmatrix}
c_0 \\
c_1 \\
\vdots \\
c_{N-1}
\end{bmatrix}, \\
\delta > 0.
\end{align*}
$$

Moreover, the matrices $P_i$ are expected to be positive semi-definite,
and the matrices $R_i$ are expected to be positive definite.

Note that when $\delta = 0$ this linear system is the Newton-KKT system
associated with a standard LQR problem.

## Motivation

When solving nonlinear optimal control problems numerically using an interior point method,
for example with the [SIP](https://github.com/joaospinto/sip) solver, the Newton-KKT
linear systems that are posed at each iterate have the form

$$
\begin{bmatrix}
P &       C^T &         G^T \\
C & -\delta I &           0 \\
G &         0 & -W - \eta I
\end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta y \\
\Delta z
\end{bmatrix} = -
\begin{bmatrix}
r_x \\
r_y \\
r_z
\end{bmatrix},
$$

where $\delta, \eta > 0$ and $W$ is a diagonal matrix with positive elements.

Note that $\Delta z$ can be eliminated via $\Delta z = (W + \eta I)^{-1}(G \Delta x + r_z)$,
resulting in

$$
\begin{bmatrix}
P + G^T (W + \eta I)^{-1} G &       C^T \\
C                           & -\delta I
\end{bmatrix}
\begin{bmatrix}
\Delta x \\
\Delta y
\end{bmatrix} = -
\begin{bmatrix}
r_x + G^T (W + \eta I)^{-1} r_z \\
r_y
\end{bmatrix}.
$$

Any component of $\Delta y$ corresponding to an equality constraint other than
the dynamics can be eliminated in the same fashion.

Importantly, note that the only constraints that have cross-stage dependencies
are the dynamics, so these eliminations preserve the stagewise nature of the problem.

This leaves us with a linear system that is a Regularized LQR problem.

## Core Algorithm

Noting that

$$
\begin{bmatrix}
P & C^T \\
C & -\delta I
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix} = -
\begin{bmatrix}
s \\
c
\end{bmatrix} \Leftrightarrow
\begin{bmatrix}
\delta P + C^T C & 0 \\
C & -\delta I
\end{bmatrix}
\begin{bmatrix}
x \\
y
\end{bmatrix} = -
\begin{bmatrix}
\delta s + C^T c \\
c
\end{bmatrix},
$$

we can solve the first equation for $x$ and recover the $y$ via the second equation.

For simplicity, we define $z = \delta s + C^T c$.

Noting that

$$
\begin{align*}
C^T C &=
\begin{bmatrix}
-I & A_0^T &       &       &        &         \\
   & B_0^T &       &       &        &         \\
   &    -I & A_1^T &       &        &         \\
   &       & B_1^T &       &        &         \\
   &       &    -I & A_2^T &        &         \\
   &       &       & B_2^T &        &         \\
   &       &       &    -I & \ddots &         \\
   &       &       &       &     -I & A_{N-1} \\
   &       &       &       &        & B_{N-1} \\
   &       &       &       &        &      -I \\
\end{bmatrix}
\begin{bmatrix}
-I  &     &     &     &     &     &    &        &         &         & \\
A_0 & B_0 &  -I &     &     &     &    &        &         &         & \\
    &     & A_1 & B_1 &  -I       &    &        &         &         & \\
    &     &     &     & A_2 & B_2 & -I &        &         &         & \\
    &     &     &     &           &    & \ddots &      -I &         & \\
    &     &     &     &           &    &        & A_{N-1} & B_{N-1} & -I
\end{bmatrix} \\
&=
\begin{bmatrix}
I + A_0^T A_0 & A_0^T B_0 &        -A_0^T &           &               &          &          &                       &                   &            \\
    B_0^T A_0 & B_0^T B_0 &        -B_0^T &           &               &          &          &                       &                   &            \\
         -A_0 &      -B_0 & I + A_1^T A_1 & A_1^T B_1 &        -A_1^T &          &          &                       &                   &            \\
              &           &     B_1^T A_1 & B_1^T B_1 &        -B_1^T &          &          &                       &                   &            \\
              &           &          -A_1 &      -B_1 & I + A_2^T A_2 &   \ddots &   \ddots &            -A_{N-2}^T &                   &            \\
              &           &               &           &        \ddots &   \ddots &   \ddots &            -B_{N-2}^T &                   &            \\
              &           &               &           &               & -A_{N-2} & -B_{N-2} & I + A_{N-1}^T A_{N-1} & A_{N-1}^T B_{N-1} & -A_{N-1}^T \\
              &           &               &           &               &          &          &     B_{N-1}^T A_{N-1} & B_{N-1}^T B_{N-1} & -B_{N-1}^T \\
              &           &               &           &               &          &          &              -A_{N-1} &          -B_{N-1} &          I
\end{bmatrix},
\end{align*}
$$

we can write

$$
\delta P + C^T C =
\begin{bmatrix}
I + A_0^T A_0 + \delta Q_0 & A_0^T B_0 + \delta M_0 &                       -A_0^T &                        &                            &          &          &                                          &                                    &                \\
  B_0^T A_0 + \delta M_0^T & B_0^T B_0 + \delta R_0 &                       -B_0^T &                        &                            &          &          &                                          &                                    &                \\
                      -A_0 &                   -B_0 &   I + A_1^T A_1 + \delta Q_1 & A_1^T B_1 + \delta M_1 &                     -A_1^T &          &          &                                          &                                    &                \\
                           &                        &     B_1^T A_1 + \delta M_1^T & B_1^T B_1 + \delta R_1 &                     -B_1^T &          &          &                                          &                                    &                \\
                           &                        &                         -A_1 &                   -B_1 & I + A_2^T A_2 + \delta Q_2 &   \ddots &   \ddots &                               -A_{N-2}^T &                                    &                \\
                           &                        &                              &                        &                     \ddots &   \ddots &   \ddots &                               -B_{N-2}^T &                                    &                \\
                           &                        &                              &                        &                            & -A_{N-2} & -B_{N-2} &   I + A_{N-1}^T A_{N-1} + \delta Q_{N-1} & A_{N-1}^T B_{N-1} + \delta M_{N-1} &     -A_{N-1}^T \\
                           &                        &                              &                        &                            &          &          &     B_{N-1}^T A_{N-1} + \delta M_{N-1}^T & B_{N-1}^T B_{N-1} + \delta R_{N-1} &     -B_{N-1}^T \\
                           &                        &                              &                        &                            &          &          &                                 -A_{N-1} &                           -B_{N-1} & I + \delta Q_N
\end{bmatrix}.
$$

In order to solve $(P + \delta C^T C) x = -z$, our strategy will be to perform
a backward pass (i.e. moving from the bottom-right corner to the top-left one),
writing

$$
u_i = K_i x_i + k_i \wedge
\begin{cases}
F_0 x_0 = -f_0, & \text{if } i = 0, \\
-A_{i-1} x_{i-1} - B_{i-1} u_{i-1} + F_{i} x_{i} = -f_{i}, & \text{if } 1 \leq i \leq N.
\end{cases}
$$

At the end of this backward pass, we will be able to recover $x_0$ and perform
a forward pass to recover the remaining $x_i$ via the equations above.

Matching the first of these equalities with the last block-row above,
we get $F_N = I + \delta Q_N$ and $f_N = z_N$.

Letting

$$
k_i^x =
\begin{cases}
z_i^x, & \text{if } i = 0, \\
z_i^x - A_{i-1}x_{i-1} -B_{i-1} u_{i-1}, & \text{if } 1 \leq i \leq N,
\end{cases} \\
$$

and noting that

$$
\begin{align*}
& \begin{bmatrix}
I + A_i^T A_i + \delta Q_i & A_i^T B_i + \delta M_i &  -A_i^T \\
  B_i^T A_i + \delta M_i^T & B_i^T B_i + \delta R_i &  -B_i^T \\
                      -A_i &                   -B_i & F_{i+1}
\end{bmatrix}
\begin{bmatrix}
x_i \\
u_i \\
x_{i+1}
\end{bmatrix}
= -\begin{bmatrix}
k_i^x \\
z_i^u \\
f_{i+1}
\end{bmatrix} \\
\Rightarrow
& \begin{bmatrix}
I + A_i^T (I - F_{i+1}^{-1}) A_i + \delta Q_i & A_i^T (I - F_{i+1}^{-1}) B_i + \delta M_i \\
  B_i^T (I - F_{i+1}^{-1}) A_i + \delta M_i^T & B_i^T (I - F_{i+1}^{-1}) B_i + \delta R_i
\end{bmatrix}
\begin{bmatrix}
x_i \\
u_i
\end{bmatrix}
= -\begin{bmatrix}
k_i^x + A_i^T F_{i+1}^{-1} f_{i+1} \\
z_i^u + B_i^T F_{i+1}^{-1} f_{i+1}
\end{bmatrix},
\end{align*}
$$

we can set

$$
\begin{align*}
K_i &= -\left( B_i^T (I - F_{i+1}^{-1}) B_i + \delta R_i \right)^{-1} \left( B_i^T (I - F_{i+1}^{-1}) A_i + \delta M_i^T \right) \\
k_i &= -\left( B_i^T (I - F_{i+1}^{-1}) B_i + \delta R_i \right)^{-1} \left( z_i^u + B_i^T F_{i+1}^{-1} f_{i+1} \right) \\
F_i &= \left( I + A_i^T (I - F_{i+1}^{-1}) A_i + \delta Q_i \right) + \left( A_i^T (I - F_{i+1}^{-1}) B_i + \delta M_i \right) K_i \\
f_i &= z_i^x + A_i^T F_{i+1}^{-1} f_{i+1} + \left( A_i^T (I - F_{i+1}^{-1}) B_i + \delta M_i \right) k_i .
\end{align*}
$$

In the interest of improving numerical stability when $\delta \rightarrow 0$, and in order to 
recover the standard LQR algorithm when $\delta = 0$, we define $V_i = \frac{1}{\delta} (F_i - I)$
and $v_i = \frac{1}{\delta}(f_i + c_i)$.

Note that $U = I + \delta V \Rightarrow I - U^{-1} = I - (I + \delta V)^{-1} = (I + \delta V)^{-1} \delta V$.
This can be easily shown:

$$
(I + \delta V)^{-1} \delta V = (I + \delta V)^{-1} (I + \delta V) - (I + \delta V)^{-1} = I - (I + \delta V)^{-1} .
$$

Moreover, note that

$$
\begin{align*}
z_i^u + B_i^T F_{i+1}^{-1} f_{i+1}
&= \delta r_i + B_i^T(c_{i+1} + F_{i+1}^{-1} f_{i+1}) \\
&= \delta r_i + B_i^T((c_{i+1} + f_{i+1}) - (I - F_{i+1}^{-1}) f_{i+1}) \\
&= \delta (r_i + B_i^T (v_{i+1} - (I + \delta V_{i+1})^{-1} V_{i+1} (\delta v_{i+1} - c_{i+1}))) \\
&= \delta (r_i + B_i^T (v_{i+1} + (I + \delta V_{i+1})^{-1} V_{i+1} (c_{i+1} - \delta v_{i+1})))
\end{align*}
$$

and that

$$
\begin{align*}
z_i^x + A_i^T F_{i+1}^{-1} f_{i+1}
&= \delta q_i - c_i + A_i^T (c_{i+1} + F_{i+1}^{-1}f_{i+1}) \\
&= \delta q_i - c_i + A_i^T ((c_{i+1} + f_{i+1}) - (I - F_{i+1}^{-1})f_{i+1}) \\
&= \delta (q_i + A_i^T (v_{i+1} - (I + \delta V_{i+1})^{-1} V_{i+1}(\delta v_{i+1} - c_{i+1}))) - c_i \\
&= \delta (q_i + A_i^T (v_{i+1} + (I + \delta V_{i+1})^{-1} V_{i+1}(c_{i+1} - \delta v_{i+1}))) - c_i .
\end{align*}
$$

The backward pass equations above can be re-written in terms of $V_i, v_i$ instead of $F_i, f_i$ as follows:

$$
\begin{align*}
K_i &= -\left( B_i^T (I + \delta V_{i+1})^{-1} V_{i+1} B_i + R_i \right)^{-1} \left( B_i^T (I + \delta V_{i+1})^{-1} V_{i+1} A_i + M_i^T \right) \\
k_i &= -\left( B_i^T (I + \delta V_{i+1})^{-1} V_{i+1} B_i + R_i \right)^{-1} \left( r_i + B_i^T v_{i+1} + (I + \delta V_{i+1})^{-1} V_{i+1} (c_{i+1} - \delta v_{i+1}) \right) \\
V_i &= A_i^T (I + \delta V_{i+1})^{-1} V_{i+1} A_i + Q_i + \left( A_i^T (I + \delta V_{i+1})^{-1} V_{i+1} B_i + M_i \right) K_i \\
v_i &= q_i + A_i^T v_{i+1} + (I + \delta V_{i+1})^{-1} V_{i+1}(c_{i+1} - \delta v_{i+1}) + \left( A_i^T (I + \delta V_{i+1})^{-1} V_{i+1} B_i + M_i \right) k_i .
\end{align*}
$$

Gathering some of the common expressions, and noting that

$$
\begin{align*}
H_i^T k_i &= H_i^T (-G_i^{-1} h_i) = (- H_i^T (G_i^{-1})^T) h_i = (-G_i^{-1} H_i)^T h_i = K_i^T h_i \\
H_i^T K_i &= H_i^T (-G_i^{-1} H_i) = (- H_i^T (G_i^{-1})^T) H_i = (-G_i^{-1} H_i)^T H_i = K_i^T H_i ,
\end{align*}
$$

this simplifies to:

$$
\begin{align*}
W_i &= (I + \delta V_{i+1})^{-1} V_{i+1} \\
G_i &= B_i^T W_i B_i + R_i \\
g_i &= v_{i+1} + W_i (c_{i+1} - \delta v_{i+1}) \\
H_i &= B_i^T W_i A_i + M_i^T \\
h_i &= r_i + B_i^T g_i \\
K_i &= -G_i^{-1} H_i \\
k_i &= -G_i^{-1} h_i \\
V_i &= A_i^T W_i A_i + Q_i + K_i^T H_i \\
v_i &= q_i + A_i^T g_i + K_i^T h_i .
\end{align*}
$$

To recover the variable $y$, we can use the identity $y = \frac{1}{\delta} (C x + c)$.
However, as before, we can re-write this in a way that is more numerically stable when
$\delta \rightarrow 0$, and that recovers the standard (dual) LQR algorithm when $\delta = 0$.

Specifically, note that

$$
\begin{align*}
y_{i + 1} &= \frac{1}{\delta} (A_i x_i + B_i u_i + c_{i+1} - x_{i+1}) \\
&= \frac{1}{\delta} \left( \left( A_i x_i + B_i u_i - f_{i+1} - F_{i+1} x_{i+1} \right) + (F_{i+1} - I) x_{i+1}  + (c_{i+1} + f_{i+1}) \right) \\
&= \frac{1}{\delta} \left( 0 + \delta V_{i+1} x_{i+1} + \delta v_{i+1} \right) \\
&= V_{i+1} x_{i+1} + v_{i+1} .
\end{align*}
$$

Moreover, since

$$
F_0 x_0 = -f_0 \Leftrightarrow (I + \delta V_0) x_0 = -(\delta v_0 - c_0) \Leftrightarrow c_0 - x_0 = \delta (V_0 x_0 + v_0),
$$

it follows that

$$
y_0 = \frac{1}{\delta} (c_0 - x_0) = V_0 x_0 + v_0.
$$
