@doc raw"""
    EulerLagrangeLoss <: NetworkLoss

The Euler–Lagrange residual loss for a Lagrangian neural network (Offen 2025, App. B).

The loss asks the learned Lagrangian to satisfy the Euler–Lagrange equations on the data, with no
solve for the acceleration:

```math
    \mathtt{loss}(\mathcal{NN}, \mathtt{input}, \mathtt{output})
        = \frac{1}{B}\sum_{b=1}^B \left\|
            \nabla_{\dot{q}}\nabla_{\dot{q}}L\,\ddot{q}_b
            + (\nabla_q\nabla_{\dot{q}}L)^T\dot{q}_b
            - \nabla_qL \right\|^2,
```

with the derivatives of ``L`` taken at sample ``b`` and
``(\nabla_q\nabla_{\dot{q}}L)_{ij} = \partial^2L/\partial{}q_i\partial\dot{q}_j``. `input` stacks
``q`` on ``\dot{q}`` and has ``2n`` rows, and `output` is ``\ddot{q}`` and has ``n``; trailing
dimensions are samples too, so an `input` of size ``(2n, s, B)`` goes with an `output` of size
``(n, s, B)``, and the mean runs over all ``sB`` samples. See
[`_euler_lagrange_residual`](@ref) for the residual of one sample.

Unlike [`LNNLoss`](@ref), the loss inverts no Hessian, so a velocity Hessian near zero does not
amplify a sample. It is not normalised: the zero Lagrangian has the loss zero on any data.

A size mismatch of `input` and `output` throws a `DimensionMismatch`, and an empty batch an
`ArgumentError`. `NaN` and `Inf` in the data propagate to the value.

# Constructor

```julia
EulerLagrangeLoss(arch)
```

where `arch` is a [`LagrangianArchitecture`](@ref).

# Functor

```julia
loss(model, ps, input, output)
loss(ps, input, output) # equivalent to the above
```
"""
struct EulerLagrangeLoss{DT <: LagrangianDerivatives} <: NetworkLoss
    derivatives::DT

    function EulerLagrangeLoss(arch::LagrangianArchitecture)
        D = LagrangianDerivatives(arch)
        new{typeof(D)}(D)
    end
end

@doc raw"""
    _euler_lagrange_residual(H, g, q̇, q̈)

The Euler–Lagrange residual of one sample (Offen 2025, App. B),

```math
    \nabla_{\dot{q}}\nabla_{\dot{q}}L\,\ddot{q} + (\nabla_q\nabla_{\dot{q}}L)^T\dot{q} - \nabla_qL,
```

where `H` is the ``2n\times{}2n`` Hessian of the Lagrangian with respect to its input
``(q, \dot{q})``, `g` the ``2n`` gradient, and `q̇` and `q̈` the ``n`` velocities and accelerations
of the sample. The transpose contracts the *first* index of
``(\nabla_q\nabla_{\dot{q}}L)_{ij} = \partial^2L/\partial{}q_i\partial\dot{q}_j`` with ``\dot{q}``,
as in [`_euler_lagrange_acceleration`](@ref).
"""
function _euler_lagrange_residual(H, g, q̇, q̈)
    n = length(q̇)
    H[(n + 1):(2n), (n + 1):(2n)] * q̈ + H[1:n, (n + 1):(2n)]' * q̇ - g[1:n]
end

function (loss::EulerLagrangeLoss)(::Union{Chain, AbstractExplicitLayer},
        ps::NetworkParameters,
        input::AbstractArray,
        output::AbstractArray)
    loss(ps, input, output)
end

function (loss::EulerLagrangeLoss)(ps::NetworkParameters,
        input::AbstractArray,
        output::AbstractArray)
    D = loss.derivatives
    n = D.dim ÷ 2
    size(input, 1) == 2n ||
        throw(DimensionMismatch("EulerLagrangeLoss: the input has $(size(input, 1)) rows, and the network takes $(2n)"))
    size(output, 1) == n ||
        throw(DimensionMismatch("EulerLagrangeLoss: the output has $(size(output, 1)) rows, and the network takes $n velocities"))
    size(input)[2:end] == size(output)[2:end] ||
        throw(DimensionMismatch("EulerLagrangeLoss: the input has the trailing size $(size(input)[2:end]), and the output $(size(output)[2:end])"))
    x = reshape(input, 2n, :)
    q̈ = reshape(output, n, :)
    batch_size = size(x, 2)
    batch_size > 0 ||
        throw(ArgumentError("EulerLagrangeLoss: the batch is empty, and a mean over no samples is undefined"))

    g = input_gradient(D, x, ps)
    H = input_hessian(D, x, ps)
    squared_residuals = [sum(abs2,
                             _euler_lagrange_residual(H[:, :, b], g[:, b],
                                 x[(n + 1):(2n), b], q̈[:, b]))
                         for b in 1:batch_size]
    sum(squared_residuals) / batch_size
end
