@doc raw"""
    lagrangian_acceleration(arch::LagrangianArchitecture)

Compute an executable expression of the acceleration that the Lagrangian `arch` predicts.

The Euler–Lagrange equations of a Lagrangian ``L(q, \dot{q})`` are

```math
    \frac{d}{dt}\frac{\partial{}L}{\partial\dot{q}} = \frac{\partial{}L}{\partial{}q},
```

which, expanded along a trajectory, reads

```math
    \nabla_{\dot{q}}\nabla_{\dot{q}}L\,\ddot{q}
        + (\nabla_q\nabla_{\dot{q}}L)^T\dot{q}
        = \nabla_qL,
```

so the acceleration is

```math
    \ddot{q} = (\nabla_{\dot{q}}\nabla_{\dot{q}}L)^{-1}
               \left( \nabla_qL - (\nabla_q\nabla_{\dot{q}}L)^T\dot{q} \right).
```

The transpose is not cosmetic. ``\nabla_q\nabla_{\dot{q}}L`` is indexed
``[i, j] = \partial^2L/\partial{}q_i\partial\dot{q}_j``, while the chain rule contracts the *first*
index with ``\dot{q}``. It disappears only for a Lagrangian whose position–velocity coupling happens
to be symmetric.

# Implementation

The gradient and the Hessian of the network with respect to its input come from
[`LagrangianDerivatives`](@ref), compiled once when `lagrangian_acceleration` is called. Trailing
input dimensions beyond the first are flattened on the way in and restored on the way out, so an
input of size ``(2n, s, B)``, which is what [`Batch`](@extref ReducedComplexityModeling.Batch)
hands the loss, yields an acceleration of size ``(n, s, B)``.
"""
function lagrangian_acceleration(arch::LagrangianArchitecture)
    D = LagrangianDerivatives(arch)
    n = dim(arch) ÷ 2

    function (input, ps)
        input_matrix = reshape(input, 2n, :)
        batch_size = size(input_matrix, 2)
        g = input_gradient(D, input_matrix, ps)
        H = input_hessian(D, input_matrix, ps)
        acceleration = reduce(hcat,
            [_euler_lagrange_acceleration(H[:, :, b], g[:, b],
                 input_matrix[(n + 1):(2n), b])
             for b in 1:batch_size])
        reshape(acceleration, n, size(input)[2:end]...)
    end
end

@doc raw"""
    _euler_lagrange_acceleration(H, g, q̇)

Solve the Euler–Lagrange equations of one sample for the acceleration.

`H` is the full ``2n\times{}2n`` Hessian of the Lagrangian with respect to its input, `g` the full
``2n`` gradient, and `q̇` the ``n`` velocity components of that same input.
"""
function _euler_lagrange_acceleration(H, g, q̇)
    n = length(q̇)
    H[(n + 1):(2n), (n + 1):(2n)] \ (g[1:n] - H[1:n, (n + 1):(2n)]' * q̇)
end

@doc raw"""
    LNNLoss <: NetworkLoss

The loss for a Lagrangian neural network.

The network output is a scalar Lagrangian ``L(q, \dot{q})``. The loss compares the acceleration its
Euler–Lagrange equations predict against the acceleration in the data:

```math
    \mathtt{loss}(\mathcal{NN}, \mathtt{input}, \mathtt{output})
        = ||\ddot{q}(\mathtt{input}) - \mathtt{output}|| \,/\, ||\mathtt{output}||,
```

where ``\mathtt{input}`` stacks ``q`` on ``\dot{q}`` and is therefore ``2n`` rows tall, and
``\mathtt{output}`` is ``\ddot{q}`` and is ``n`` rows tall. See [`lagrangian_acceleration`](@ref)
for ``\ddot{q}``.

# Constructor

This can be called with a [`LagrangianArchitecture`](@ref) as its only argument:
```julia
LNNLoss(arch)
```

# Functor

```julia
loss(model, ps, input, output)
loss(ps, input, output) # equivalent to the above
```
"""
struct LNNLoss{FT} <: NetworkLoss
    acceleration::FT
end

function LNNLoss(arch::LagrangianArchitecture)
    LNNLoss(lagrangian_acceleration(arch))
end

AbstractNeuralNetworks.NetworkLoss(arch::LagrangianArchitecture) = LNNLoss(arch)

function (loss::LNNLoss)(::Union{Chain, AbstractExplicitLayer},
        ps::NetworkParameters,
        input::AbstractArray,
        output::AbstractArray)
    loss(ps, input, output)
end

function (loss::LNNLoss)(ps::NetworkParameters,
        input::AbstractArray,
        output::AbstractArray)
    norm(loss.acceleration(input, ps) - output) / norm(output)
end
