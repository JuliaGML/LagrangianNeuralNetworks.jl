# The backward error analysis of a discrete model: the map from the inverse modified Lagrangian of
# the midpoint rule to the true Lagrangian of the flow it reproduces, to second order in the
# timestep (Ober-Blöbaum & Offen 2023, (11); Lishkova 2023, (4)).

@doc raw"""
    modified_lagrangian(model, Δt)

The continuous Lagrangian whose exact flow the discrete model of `model` reproduces to
``O(\Delta{}t^4)``, as the function `(q, q̇) ↦ L + (Δt²/24)·[(L_q − L_qq̇·q̇)²/L_q̇q̇ − L_qq·q̇²]` of
two vectors of one entry.

This is the second-order backward error analysis of Ober-Blöbaum & Offen 2023, (11), in the
direction that recovers the true Lagrangian. The midpoint rule reproduces the exact flow of its
*inverse* modified Lagrangian, which for a continuous network is the network itself and for a
[`DiscreteLagrangianNetwork`](@ref) is
``L_{im}(q, \dot{q}) = L_d(q - \Delta{}t\dot{q}/2, q + \Delta{}t\dot{q}/2)/\Delta{}t``
(Lishkova 2023, (4)). The formula maps that inverse modified Lagrangian to the true one; its plus
sign gives a frequency error of ``O(\Delta{}t^4)`` on the harmonic oscillator and the opposite sign
``O(\Delta{}t^2)``.

For a continuous network the derivatives are the compiled input derivatives of the network. For a
[`DiscreteLagrangianArchitecture`](@ref) they are the chain rule through the midpoint, with `H` the
blocks of the network's own input Hessian at `[q − Δt q̇/2; q + Δt q̇/2]`:

```math
    L_q = \frac{D_1L_d + D_2L_d}{\Delta{}t}, \qquad
    L_{\dot{q}} = \frac{D_2L_d - D_1L_d}{2}, \qquad
    L_{qq} = \frac{H_{11} + H_{12} + H_{21} + H_{22}}{\Delta{}t},
```

```math
    L_{q\dot{q}} = \frac{H_{12} + H_{22} - H_{11} - H_{21}}{2}, \qquad
    L_{\dot{q}\dot{q}} = \frac{\Delta{}t}{4}\left(H_{11} - H_{12} - H_{21} + H_{22}\right).
```

The analysis is one-dimensional: a model of more than one degree of freedom throws an
`ArgumentError`, and so does a timestep that is not positive and finite in the element type of the
parameters. A `q` or a `q̇` of another length throws a `DimensionMismatch`. A zero
``L_{\dot{q}\dot{q}}`` gives the IEEE value of the quotient, and no exception. The value has the
element type of the network's parameters.
"""
function modified_lagrangian(
        model::AbstractNeuralNetworks.NeuralNetwork{<:LagrangianArchitecture}, Δt::Real)
    Δt = _check_scalar_analysis(model, Δt)
    arch = model.architecture
    ps = model.params
    T = parameter_eltype(ps)
    δ = _check_timestep(T(Δt))
    D = LagrangianDerivatives(arch)
    (q::AbstractVector, q̇::AbstractVector) -> begin
        _check_analysis_pair(q, q̇)
        x = reshape(vcat(q, q̇), 2, 1)
        g = input_gradient(D, x, ps)
        H = input_hessian(D, x, ps)
        _backward_error_value(only(model(x, ps)), g[1, 1], H[1, 1, 1], H[1, 2, 1],
            H[2, 2, 1], q̇[1], δ, T)
    end
end

function modified_lagrangian(
        model::AbstractNeuralNetworks.NeuralNetwork{<:DiscreteLagrangianArchitecture},
        Δt::Real)
    Δt = _check_scalar_analysis(model, Δt)
    T = parameter_eltype(model.params)
    δ = _check_timestep(T(Δt))
    D = LagrangianDerivatives(model.architecture)
    ps = model.params
    L = inverse_modified_lagrangian(model, Δt)
    (q::AbstractVector, q̇::AbstractVector) -> begin
        _check_analysis_pair(q, q̇)
        x = reshape(vcat(q - δ * q̇ / 2, q + δ * q̇ / 2), 2, 1)
        g = input_gradient(D, x, ps)
        H = input_hessian(D, x, ps)
        H₁₁, H₁₂, H₂₁, H₂₂ = H[1, 1, 1], H[1, 2, 1], H[2, 1, 1], H[2, 2, 1]
        _backward_error_value(L(q, q̇),
            (g[1, 1] + g[2, 1]) / δ,
            (H₁₁ + H₁₂ + H₂₁ + H₂₂) / δ,
            (H₁₂ + H₂₂ - H₁₁ - H₂₁) / 2,
            (δ / 4) * (H₁₁ - H₁₂ - H₂₁ + H₂₂),
            q̇[1], δ, T)
    end
end

# The timestep of a one-dimensional analysis: the guards of the other model constructors of the
# package, and the degree-of-freedom count this formula is written for.
function _check_scalar_analysis(model::AbstractNeuralNetworks.NeuralNetwork, Δt::Real)
    Δt = _check_timestep(Δt)
    n = dim(model.architecture) ÷ 2
    n == 1 || throw(ArgumentError(
        "the backward error analysis is one-dimensional, and the model has $n degrees of freedom"))
    Δt
end

function _check_analysis_pair(q::AbstractVector, q̇::AbstractVector)
    length(q) == 1 || throw(DimensionMismatch(
        "q has $(length(q)) entries, and the analysis is one-dimensional"))
    length(q̇) == 1 || throw(DimensionMismatch(
        "q̇ has $(length(q̇)) entries, and the analysis is one-dimensional"))
    nothing
end

# The corrected Lagrangian of one point, in the element type of the parameters. `L` is the inverse
# modified Lagrangian there; the bracket is Ober-Blöbaum & Offen 2023, (11).
function _backward_error_value(L, Lq, Lqq, Lqq̇, Lq̇q̇, q̇, δ, T)
    T(L + (δ^2 / 24) * ((Lq - Lqq̇ * q̇)^2 / Lq̇q̇ - Lqq * q̇^2))
end
