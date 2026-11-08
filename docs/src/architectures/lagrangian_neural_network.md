# Lagrangian Neural Network

A *Lagrangian neural network* learns a scalar Lagrangian ``L(q, \dot{q})`` instead of the dynamics
themselves. The dynamics are then whatever that Lagrangian implies, through the Euler–Lagrange
equations

```math
    \frac{d}{dt}\frac{\partial{}L}{\partial\dot{q}} = \frac{\partial{}L}{\partial{}q}.
```

This is the Lagrangian counterpart of the Hamiltonian neural network of
HamiltonianNeuralNetworks.jl: there the network is a Hamiltonian and the dynamics come from a
Poisson tensor, here the network is a Lagrangian and the dynamics come from a variational
principle.

Each section below describes one method, cites its source, and documents its functions. The
[tutorial](@ref lnn_tutorial) trains four models of the harmonic oscillator with them and compares
the results.

## The Architecture

[`LagrangianNeuralNetwork`](@ref LagrangianNeuralNetworks.LagrangianNeuralNetwork) is the network
of [cranmer2020](@citet): a fully connected network from the state ``x = [q; \dot{q}]`` to a scalar,
which is read as the Lagrangian. Its default activation is
[`softplus`](@ref LagrangianNeuralNetworks.softplus), which the source chose from a search over
activations because a network of ReLUs has a zero second derivative, and the dynamics need the
second. [hamzaogullari2026](@citet) find GeLU the best activation on their pendulum benchmarks; the
package does not offer it yet.

The network is unstructured: any scalar function of the state is a candidate. The structured
architecture ``L = \frac{1}{2}\dot{q}^TM(q)\dot{q} - V(q)`` of [lutter2019](@citet), with the mass
matrix ``M = \hat{L}\hat{L}^T`` from a Cholesky factor of positive diagonal, excludes singular
velocity Hessians by construction; the package does not offer it yet.

```@docs
LagrangianNeuralNetworks.LagrangianArchitecture
LagrangianNeuralNetworks.LagrangianNeuralNetwork
LagrangianNeuralNetworks.softplus
```

## The Loss

Expanding the Euler–Lagrange equations along a trajectory gives

```math
    \nabla_{\dot{q}}\nabla_{\dot{q}}L\,\ddot{q}
        + (\nabla_q\nabla_{\dot{q}}L)^T\dot{q}
        = \nabla_qL,
```

so the acceleration the learned Lagrangian predicts is

```math
    \ddot{q} = (\nabla_{\dot{q}}\nabla_{\dot{q}}L)^{-1}
               \left( \nabla_qL - (\nabla_q\nabla_{\dot{q}}L)^T\dot{q} \right),
```

which is what [`LagrangianNeuralNetworks.lagrangian_acceleration`](@ref) computes and what
[`LagrangianNeuralNetworks.LNNLoss`](@ref) compares against the data. This is the loss of
[cranmer2020](@citet).

The transpose is not cosmetic. ``\nabla_q\nabla_{\dot{q}}L`` is indexed
``[i, j] = \partial^2L/\partial{}q_i\partial\dot{q}_j``, while the chain rule contracts the *first*
index with ``\dot{q}``. It disappears only for a Lagrangian whose position–velocity coupling happens
to be symmetric, which is why the test for this asserts a closed form whose coupling deliberately is
not.

```@docs
LagrangianNeuralNetworks.LNNLoss
LagrangianNeuralNetworks.lagrangian_acceleration
LagrangianNeuralNetworks._euler_lagrange_acceleration
```

## Why the Derivatives Are Symbolic

The loss needs the *second* derivative of the network with respect to its input, and then its own
derivative with respect to the network parameters. Taking the inner one with `Zygote` inside the
loss does not work: the parameter gradient then fails with
`MethodError: no method matching getindex(::IdDict{Any, Any})`. The gradient and Hessian are
therefore built as symbolic expressions with
[`SymbolicNeuralNetworks`](https://github.com/JuliaGNI/SymbolicNeuralNetworks.jl) and compiled, so
that only one differentiation is left for `Zygote` to do. This is the same route that
`hamiltonian_vector_field` of HamiltonianNeuralNetworks.jl takes, and for the same reason.

```@docs
LagrangianNeuralNetworks.LagrangianDerivatives
LagrangianNeuralNetworks.input_gradient
LagrangianNeuralNetworks.input_hessian
```

## The Euler–Lagrange Residual

[`EulerLagrangeLoss`](@ref LagrangianNeuralNetworks.EulerLagrangeLoss) asks the learned
Lagrangian to satisfy the Euler–Lagrange equations on the data, as the residual of the first
equation above, with no solve against the velocity Hessian [offen2025](@cite). A velocity Hessian
near zero then does not amplify a sample. The price is that the zero Lagrangian, and every null
Lagrangian, has the loss zero: the loss is used together with a normalisation and a scale
condition, as the next section describes.

```@docs
LagrangianNeuralNetworks.EulerLagrangeLoss
LagrangianNeuralNetworks._euler_lagrange_residual
```

## Normalisation

A Lagrangian is determined by its dynamics only up to a constant, a total time derivative and a
factor (see [the non-uniqueness of a Lagrangian](@ref "The Non-Uniqueness of a Lagrangian")).
[offen2025](@citet) fixes the value and the momentum of the Lagrangian at one base point,
Prop. 3.1, which removes the constant and the linear gauge and which
[`NormalisedLagrangian`](@ref LagrangianNeuralNetworks.NormalisedLagrangian) does for every value
of the parameters, and fixes the factor through the determinant of the velocity Hessian at that
point, (B.1). [`ScaleCondition`](@ref LagrangianNeuralNetworks.ScaleCondition) is that condition as
a penalty, and [`DiscreteScaleCondition`](@ref LagrangianNeuralNetworks.DiscreteScaleCondition)
the condition (B.2) on the mixed Hessian of a discrete Lagrangian; [oberbloebaum2023](@citet) fix
the scale of a discrete Lagrangian by a condition of this kind, (13)–(15).

```@docs
LagrangianNeuralNetworks.NormalisedLagrangian
LagrangianNeuralNetworks.ScaleCondition
LagrangianNeuralNetworks.DiscreteScaleCondition
```

## Regularity

A Lagrangian whose velocity Hessian is singular does not determine an acceleration, and a discrete
Lagrangian whose mixed Hessian is singular does not determine the next position.
[offen2023](@citet) penalise the norm of the inverse of that Hessian, and [offen2024](@citet) tame
the penalty so that it vanishes wherever the inverse is small enough, (34);
[lishkova2023](@citet) penalise a small determinant through a sigmoid.
[`VelocityHessianRegulariser`](@ref LagrangianNeuralNetworks.VelocityHessianRegulariser) is the
penalty of a continuous Lagrangian, and
[`MixedHessianRegulariser`](@ref LagrangianNeuralNetworks.MixedHessianRegulariser) that of the
midpoint discrete Lagrangian, in the three forms. [`RegularisedLoss`](@ref
LagrangianNeuralNetworks.RegularisedLoss) adds weighted penalties, these or the scale conditions,
to any loss of the package.

```@docs
LagrangianNeuralNetworks.RegularisedLoss
LagrangianNeuralNetworks.VelocityHessianRegulariser
LagrangianNeuralNetworks.MixedHessianRegulariser
LagrangianNeuralNetworks._mixed_hessian
LagrangianNeuralNetworks._regularity_penalty
LagrangianNeuralNetworks._check_form
```

## Training on Positions Alone

[`LagrangianNeuralNetworks.LNNLoss`](@ref) needs ``\ddot{q}`` in the data. When only positions are
available, [`LagrangianNeuralNetworks.VariationalMidpointLoss`](@ref) trains the same architecture
through the discrete Euler–Lagrange equations of the *midpoint discrete Lagrangian*

```math
    L_d(q_n, q_{n+1}) = \Delta{}t\,
        L\!\left( \frac{q_n + q_{n+1}}{2}, \frac{q_{n+1} - q_n}{\Delta{}t} \right),
```

which ask that ``D_2L_d(q_n, q_{n+1}) + D_1L_d(q_{n+1}, q_{n+2}) = 0`` along the data. The loss is
the mean squared residual of these equations, (16) of [oberbloebaum2023](@citet). The residual
needs only the first derivative of the network, because the chain rule through the midpoint is
written out rather than taken with a nested `Zygote.gradient`; its default regulariser,
[`LagrangianNeuralNetworks.MixedHessianRegulariser`](@ref), needs the second.

```@docs
LagrangianNeuralNetworks.VariationalMidpointLoss
LagrangianNeuralNetworks._discrete_lagrangian_derivatives
```

## The Discrete Lagrangian Network

[lishkova2023](@citet) learn the discrete Lagrangian ``L_d(q_0, q_1)`` itself, as a network of two
consecutive positions, (14), with the squared discrete Euler–Lagrange residual as the loss.
[`DiscreteLagrangianNetwork`](@ref LagrangianNeuralNetworks.DiscreteLagrangianNetwork) is that
network and [`DiscreteEulerLagrangeLoss`](@ref LagrangianNeuralNetworks.DiscreteEulerLagrangeLoss)
that loss. The continuous Lagrangian a discrete one stands for is its inverse modified Lagrangian,
(4) of the source, which
[`inverse_modified_lagrangian`](@ref LagrangianNeuralNetworks.inverse_modified_lagrangian) reads
off the network.

```@docs
LagrangianNeuralNetworks.DiscreteLagrangianArchitecture
LagrangianNeuralNetworks.DiscreteLagrangianNetwork
LagrangianNeuralNetworks.DiscreteEulerLagrangeLoss
LagrangianNeuralNetworks.inverse_modified_lagrangian
```

## The Variational Integrator

A learned Lagrangian is rolled out by the variational integrator of its discrete Lagrangian: each
step solves the discrete Euler–Lagrange equations for the next position by Newton's method
[offen2023](@cite). [`DiscreteLagrangian`](@ref LagrangianNeuralNetworks.DiscreteLagrangian) is the
discrete model, the midpoint rule of a continuous network or a discrete network itself, and
[`variational_rollout`](@ref LagrangianNeuralNetworks.variational_rollout) the rollout.

```@docs
LagrangianNeuralNetworks.DiscreteLagrangian
LagrangianNeuralNetworks.variational_rollout
```

## Backward Error Analysis

A model trained on positions learns the Lagrangian whose midpoint rule reproduces the data, its
inverse modified Lagrangian, which differs from the true Lagrangian of the data at
``O(\Delta{}t^2)``. [oberbloebaum2023](@citet) give the map from the one to the other to second
order in the timestep, (11), and [lishkova2023](@citet) the inverse modified Lagrangian of a
discrete network, (4). [`modified_lagrangian`](@ref LagrangianNeuralNetworks.modified_lagrangian)
applies the map, and removes that offset.

```@docs
LagrangianNeuralNetworks.modified_lagrangian
```

## The Non-Uniqueness of a Lagrangian

The data fix the equations of motion, not the Lagrangian. If ``L`` gives the dynamics, so does

```math
    \tilde{L} = \rho\,L + c + \frac{d}{dt}F(q)
```

for every scale factor ``\rho \neq 0``, every constant ``c`` and every total time derivative of a
function ``F`` of the positions, and in one dimension even Lagrangians that are not of this form,
inequivalent ones, give the same equations of motion. A learned Lagrangian is therefore not
compared with the true one: a fit of the one to the other can be poor where the dynamics are
learned well. The normalisation of [offen2025](@citet) removes the constant, the linear gauge and
the scale, but a model without it is one representative of a whole class.

The package measures a learned model by what does not depend on the representative:

- [`acceleration_error`](@ref LagrangianNeuralNetworks.acceleration_error), the relative error of
  the acceleration that the model predicts on a grid of states, §7.3 of [offen2025](@cite);
- the [`energy`](@ref LagrangianNeuralNetworks.energy) ``E = \dot{q}^T\partial_{\dot{q}}L - L``,
  which becomes ``\rho E - c`` under the gauge, so that its level sets do not change, and
  [`energy_alignment`](@ref LagrangianNeuralNetworks.energy_alignment), the mean
  ``|\sin\theta|`` between the gradients of the learned and the true energy, the measure ``\nu`` of
  §3.1 of [oberbloebaum2023](@cite);
- [`trajectory_error`](@ref LagrangianNeuralNetworks.trajectory_error), the largest distance
  between a rollout of the model and a reference trajectory.

```@docs
LagrangianNeuralNetworks.LagrangianEnergy
LagrangianNeuralNetworks.energy
LagrangianNeuralNetworks.acceleration_error
LagrangianNeuralNetworks.energy_alignment
LagrangianNeuralNetworks.trajectory_error
```
