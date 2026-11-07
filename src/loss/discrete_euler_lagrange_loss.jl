# The penalty of a discrete Lagrangian network: the state is the pair itself, so no midpoint is
# taken and the Hessian is the network's own.
struct DiscreteMixedHessianRegulariser{DT <: LagrangianDerivatives, TT <: Real}
    derivatives::DT
    timestep::TT
    form::Symbol
end

@doc raw"""
    MixedHessianRegulariser(arch::DiscreteLagrangianArchitecture, Δt; form = :tamed)

The regularity penalty of a discrete Lagrangian network, the counterpart for a
[`DiscreteLagrangianArchitecture`](@ref) of the penalty of the midpoint discrete Lagrangian: the
mean over the pairs ``[q_0; q_1]`` of a penalty of

```math
    \Lambda = -\Delta{}t\,\frac{\partial^2L_d}{\partial{}q_0\partial{}q_1},
```

read from the input Hessian ``H`` of the network itself, with
``D_1D_2L_d[i, j] = \partial^2L_d/\partial{}a_i\partial{}b_j = H[i, n + j]``. It takes the three
forms `:inverse`, `:tamed` and `:sigmoid` of [`_regularity_penalty`](@ref), over the same
threshold: ``\Lambda`` tends to the velocity Hessian ``\nabla_{\dot{q}}\nabla_{\dot{q}}L`` of the
continuous Lagrangian that the network stands for as ``\Delta{}t \to 0``, so a threshold on it
means the same as one on the velocity Hessian.

A singular ``\Lambda`` gives `Inf` for `:inverse`, `1` for `:tamed` and `1/2` for `:sigmoid`.

The result `r` is called as `r(ps, input)`, with `input` the pairs ``[q_0; q_1]``: ``2n`` rows, and
any trailing dimensions, which count as samples. The value has the element type of `input`, and
`Δt` is converted to it. A timestep that is not positive and finite and an unknown `form` throw an
`ArgumentError`, and an `input` of another row count than ``2n`` a `DimensionMismatch`.
"""
function MixedHessianRegulariser(
        arch::DiscreteLagrangianArchitecture, Δt::Real; form = :tamed)
    _check_timestep(Δt)
    _check_form(form)
    DiscreteMixedHessianRegulariser(LagrangianDerivatives(arch), Δt, form)
end

function (r::DiscreteMixedHessianRegulariser)(ps::NetworkParameters, input::AbstractArray)
    x = _input_columns(input, r.derivatives)
    n = r.derivatives.dim ÷ 2
    Δt = eltype(x)(r.timestep)
    H = input_hessian(r.derivatives, x, ps)
    batch_size = size(x, 2)
    penalties = [_regularity_penalty(-Δt * H[1:n, (n + 1):(2n), b], r.form)
                 for b in 1:batch_size]
    sum(penalties) / batch_size
end

@doc raw"""
    DiscreteEulerLagrangeLoss(arch, Δt;
        regulariser = MixedHessianRegulariser(arch, Δt; form = :tamed), weight = 1)

The loss that trains a [`DiscreteLagrangianNetwork`](@ref) on positions alone, through the discrete
Euler–Lagrange equations of the network itself:

```math
    D_2L_d(q_{k-1}, q_k) + D_1L_d(q_k, q_{k+1}) = 0 .
```

It is the mean over the samples of the squared residual (Lishkova 2023, (14)), plus a weighted
regularity penalty:

```math
    \frac{1}{B}\sum_{b=1}^B \left\| D_2L_d(q_{k-1}, q_k) + D_1L_d(q_k, q_{k+1}) \right\|^2
        + w\,r(ps, [q_{k-1}; q_k]),
```

with ``D_1L_d`` and ``D_2L_d`` the first and the last ``n`` entries of the input gradient of the
network, and ``w`` = `weight`. The residual alone does not exclude a null discrete Lagrangian: a
constant, and ``g(q_0) - g(q_1)`` with ``g`` linear, have a residual of zero on any data. The
default penalty, [`MixedHessianRegulariser`](@ref) in the form `:tamed`, is 1 where the discrete
Lagrangian is degenerate, as it is for those; its ``\Lambda = -\Delta{}t\,D_1D_2L_d`` tends to the
velocity Hessian of the continuous Lagrangian as ``\Delta{}t \to 0``.

The residual needs only the first derivative of the network with respect to its input, which comes
from the compiled symbolic expression of [`LagrangianDerivatives`](@ref), and the regulariser
compiles its own Hessian.

# Constructor

`arch` is a [`DiscreteLagrangianArchitecture`](@ref), `Δt` its timestep, `regulariser` a callable
`r(ps, input)` and `weight` its weight ``w``. The timestep is held by the regulariser, which the
constructor builds from it; the loss itself refuses a timestep that is not positive and finite, and
a weight that is negative, `NaN` or `Inf`, with an `ArgumentError`. With `weight = 0` the
regulariser is not evaluated, and the loss is the mean squared residual alone.

# Functor

```julia
loss(model, ps, input, output)
loss(ps, input, output) # equivalent to the above
```

`input` stacks ``q_{k-1}`` on ``q_k`` and so has ``2n`` rows; `output` is ``q_{k+1}`` and has ``n``.
Trailing dimensions are samples. Another row count throws a `DimensionMismatch`. The value has the
element type of the data.
"""
struct DiscreteEulerLagrangeLoss{DT <: LagrangianDerivatives, RT, WT <: Real} <: NetworkLoss
    derivatives::DT
    regulariser::RT
    weight::WT

    function DiscreteEulerLagrangeLoss(arch::DiscreteLagrangianArchitecture, Δt::Real;
            regulariser = MixedHessianRegulariser(arch, Δt; form = :tamed),
            weight::Real = 1)
        _check_timestep(Δt)
        _check_weight(weight)
        D = LagrangianDerivatives(arch; hessian = false)
        new{typeof(D), typeof(regulariser), typeof(weight)}(D, regulariser, weight)
    end
end

function (loss::DiscreteEulerLagrangeLoss)(::Union{Chain, AbstractExplicitLayer},
        ps::NetworkParameters, input::AbstractArray, output::AbstractArray)
    loss(ps, input, output)
end

function (loss::DiscreteEulerLagrangeLoss)(
        ps::NetworkParameters, input::AbstractArray, output::AbstractArray)
    x = _input_columns(input, loss.derivatives)
    n = size(x, 1) ÷ 2
    size(output, 1) == n ||
        throw(DimensionMismatch("the output has $(size(output, 1)) rows, and the architecture needs $n"))
    qₖ₋₁ = x[1:n, :]
    qₖ = x[(n + 1):(2n), :]
    qₖ₊₁ = reshape(output, n, :)

    # The last `n` entries of the input gradient at the pair `(q_{k-1}, q_k)` are
    # `D₂L_d(q_{k-1}, q_k)`, and the first `n` at the pair `(q_k, q_{k+1})` are
    # `D₁L_d(q_k, q_{k+1})`.
    g₂ = input_gradient(loss.derivatives, vcat(qₖ₋₁, qₖ), ps)
    g₁ = input_gradient(loss.derivatives, vcat(qₖ, qₖ₊₁), ps)
    D₂ = g₂[(n + 1):(2n), :]
    D₁ = g₁[1:n, :]

    residual = sum(abs2, D₂ + D₁) / size(qₖ₊₁, 2)
    # A regulariser of weight 0 is not evaluated, so that a penalty of `Inf` or `NaN` does not make
    # the loss `NaN`.
    iszero(loss.weight) && return residual
    residual + eltype(x)(loss.weight) * loss.regulariser(ps, input)
end
