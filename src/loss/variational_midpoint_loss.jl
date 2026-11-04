@doc raw"""
    VariationalMidpointLoss <: NetworkLoss

The loss that trains a Lagrangian neural network on positions alone, through the discrete
Euler–Lagrange equations of the midpoint discrete Lagrangian.

It needs three consecutive positions and the timestep between them — no velocities and no
accelerations.

The midpoint discrete Lagrangian is

```math
    L_d(q_n, q_{n+1}) = \Delta{}t\,
        L\!\left( \frac{q_n + q_{n+1}}{2}, \frac{q_{n+1} - q_n}{\Delta{}t} \right),
```

and the discrete Euler–Lagrange equations ask that

```math
    D_2L_d(q_n, q_{n+1}) + D_1L_d(q_{n+1}, q_{n+2}) = 0 .
```

The loss is the mean over the samples of the squared residual (Ober-Blöbaum & Offen 2023, (16)),
plus a weighted regularity penalty:

```math
    \frac{1}{B}\sum_{b=1}^B \left\| D_2L_d(q_n, q_{n+1}) + D_1L_d(q_{n+1}, q_{n+2}) \right\|^2
        + w\,r(ps, [q_n; q_{n+1}]).
```

The residual alone does not exclude a null Lagrangian: a constant, or ``g(q)\cdot\dot{q}`` with
``g = \nabla\varphi`` and ``\varphi`` quadratic, has a discrete residual of zero on any data. The
default penalty, [`MixedHessianRegulariser`](@ref) in the form `:tamed`, is 1 where the discrete
Lagrangian is degenerate, as it is for those. The relative normalisation of the source is not
applied.

# Implementation

The two slot derivatives are written out with the chain rule rather than taken with a nested
`Zygote.gradient`:

```math
    D_1L_d = \frac{\Delta{}t}{2}\nabla_qL - \nabla_{\dot{q}}L, \qquad
    D_2L_d = \frac{\Delta{}t}{2}\nabla_qL + \nabla_{\dot{q}}L,
```

both evaluated at the midpoint. The residual needs only the *first* derivative of the network with
respect to its input, and that comes from the compiled symbolic expression of
[`LagrangianDerivatives`](@ref), exactly as that of `hamiltonian_vector_field` of
HamiltonianNeuralNetworks.jl does; the regulariser compiles its own derivatives. Written the other
way — a `Zygote.gradient` inside the loss — the parameter gradient fails with
`MethodError: no method matching getindex(::IdDict{Any, Any})`.

# Constructor

```julia
VariationalMidpointLoss(arch, timestep;
    regulariser = MixedHessianRegulariser(arch, timestep; form = :tamed), weight = 1)
```

where `arch` is a [`LagrangianArchitecture`](@ref), `regulariser` a callable `r(ps, input)` and
`weight` its weight ``w``. A timestep that is not positive and finite, and a weight that is
negative, `NaN` or `Inf`, throw an `ArgumentError`. With `weight = 0` the regulariser is not
evaluated, and the loss is the mean squared residual alone.

# Functor

```julia
loss(model, ps, input, output)
loss(ps, input, output) # equivalent to the above
```

`input` stacks ``q_n`` on ``q_{n+1}`` and so has ``2n`` rows; `output` is ``q_{n+2}`` and has ``n``.
Trailing dimensions are samples. Another row count throws a `DimensionMismatch`. The timestep and
the weight are converted to the element type of the data, which the value has.
"""
struct VariationalMidpointLoss{DT <: LagrangianDerivatives, T, RT, WT <: Real} <:
       NetworkLoss
    derivatives::DT
    timestep::T
    regulariser::RT
    weight::WT

    function VariationalMidpointLoss(arch::LagrangianArchitecture, timestep::Real;
            regulariser = MixedHessianRegulariser(arch, timestep; form = :tamed),
            weight::Real = 1)
        _check_timestep(timestep)
        _check_weight(weight)
        D = LagrangianDerivatives(arch; hessian = false)
        new{typeof(D), typeof(timestep), typeof(regulariser), typeof(weight)}(
            D, timestep, regulariser, weight)
    end
end

@doc raw"""
    _discrete_lagrangian_derivatives(loss, ps, qa, qb)

Return `(D₁Lᵈ, D₂Lᵈ)` of the midpoint discrete Lagrangian on the pair `(qa, qb)`, one column per
sample.
"""
function _discrete_lagrangian_derivatives(loss::VariationalMidpointLoss, ps,
        qa::AbstractMatrix, qb::AbstractMatrix)
    n = size(qa, 1)
    # The timestep is converted to the data's element type rather than promoted against it: a
    # `Float64` timestep must not widen the loss of a `Float32` network.
    Δt = eltype(qa)(loss.timestep)
    midpoint = vcat((qa + qb) / 2, (qb - qa) / Δt)
    g = input_gradient(loss.derivatives, midpoint, ps)
    ∇qL = g[1:n, :]
    ∇q̇L = g[(n + 1):(2n), :]
    (Δt / 2) * ∇qL - ∇q̇L, (Δt / 2) * ∇qL + ∇q̇L
end

function (loss::VariationalMidpointLoss)(::Union{Chain, AbstractExplicitLayer},
        ps::NetworkParameters, input::AbstractArray, output::AbstractArray)
    loss(ps, input, output)
end

function (loss::VariationalMidpointLoss)(
        ps::NetworkParameters, input::AbstractArray, output::AbstractArray)
    n = loss.derivatives.dim ÷ 2
    size(input, 1) == 2n ||
        throw(DimensionMismatch("the input has $(size(input, 1)) rows, and the architecture needs $(2n)"))
    size(output, 1) == n ||
        throw(DimensionMismatch("the output has $(size(output, 1)) rows, and the architecture needs $n"))
    qₙ = reshape(selectdim(input, 1, 1:n), n, :)
    qₙ₊₁ = reshape(selectdim(input, 1, (n + 1):(2n)), n, :)
    qₙ₊₂ = reshape(output, n, :)

    _, D₂ = _discrete_lagrangian_derivatives(loss, ps, qₙ, qₙ₊₁)
    D₁, _ = _discrete_lagrangian_derivatives(loss, ps, qₙ₊₁, qₙ₊₂)

    residual = sum(abs2, D₂ + D₁) / size(qₙ₊₂, 2)
    # A regulariser of weight 0 is not evaluated, so that a penalty of `Inf` or `NaN` does not make
    # the loss `NaN`.
    iszero(loss.weight) && return residual
    residual + eltype(qₙ)(loss.weight) * loss.regulariser(ps, input)
end
