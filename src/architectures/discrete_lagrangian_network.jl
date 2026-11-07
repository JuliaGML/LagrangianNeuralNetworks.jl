@doc raw"""
    DiscreteLagrangianArchitecture <: Architecture

The supertype of an architecture whose network is a discrete Lagrangian ``L_d(q_0, q_1)``, a scalar
function of two consecutive positions (Lishkova 2023, (14)).

It is no [`LagrangianArchitecture`](@ref): its input `[q_0; q_1]` stacks two positions, not a
position and a velocity, so a loss that reads a position and a velocity, such as [`LNNLoss`](@ref),
would take the second position for a velocity. A subtype needs two methods,
`AbstractNeuralNetworks.dim(arch)`, the input dimension ``2n``, and `Chain(arch)`, a chain of ``2n``
inputs with a scalar output. The subtype of the package is
[`DiscreteLagrangianNetwork`](@ref), which [`DiscreteEulerLagrangeLoss`](@ref) trains and from which
[`inverse_modified_lagrangian`](@ref) reads a continuous Lagrangian.
"""
abstract type DiscreteLagrangianArchitecture <: Architecture end

@doc raw"""
    DiscreteLagrangianNetwork(n; width = 2n, nhidden = 1, activation = softplus)

The discrete Lagrangian ``L_d(q_0, q_1)`` of Lishkova 2023, (14), as a network: a `Dense` layer of
`width` neurons, `nhidden` more of them, and a bias-free `Linear` output layer. Its input
`[q_0; q_1]` stacks two positions of ``n`` entries, and its output is a scalar.

The network is a [`DiscreteLagrangianArchitecture`](@ref), not a [`LagrangianArchitecture`](@ref).
[`DiscreteEulerLagrangeLoss`](@ref) trains it on positions alone, through the discrete
Euler–Lagrange equations, and [`inverse_modified_lagrangian`](@ref) reads a continuous Lagrangian
off it. The layers are those of [`LagrangianNeuralNetwork`](@ref), with ``2n`` inputs in place of
`dimin`, and the default activation is [`softplus`](@ref).

`n`, `width` and `nhidden` are stored as `Int`. `n < 1`, `width < 1` and `nhidden < 0` throw an
`ArgumentError`.
"""
struct DiscreteLagrangianNetwork{AT} <: DiscreteLagrangianArchitecture
    n::Int
    width::Int
    nhidden::Int
    act::AT

    function DiscreteLagrangianNetwork(
            n::Integer; width = 2n, nhidden = 1, activation = softplus)
        n < 1 && throw(ArgumentError("n must be positive, got $n"))
        width < 1 && throw(ArgumentError("width must be positive, got $width"))
        nhidden < 0 && throw(ArgumentError("nhidden must be non-negative, got $nhidden"))
        new{typeof(activation)}(Int(n), Int(width), Int(nhidden), activation)
    end
end

@inline AbstractNeuralNetworks.dim(arch::DiscreteLagrangianNetwork) = 2 * arch.n

function Chain(nn::DiscreteLagrangianNetwork)
    Chain(LagrangianNeuralNetwork(
        dim(nn); width = nn.width, nhidden = nn.nhidden, activation = nn.act))
end

@doc raw"""
    inverse_modified_lagrangian(nn::NeuralNetwork, Δt)
    inverse_modified_lagrangian(Ld::Function, Δt)

The continuous Lagrangian that a discrete Lagrangian stands for, the inverse modified Lagrangian of
Lishkova 2023, (4):

```math
    L(q, \dot{q}) = \frac{1}{\Delta{}t}\,
        L_d\!\left(q - \frac{\Delta{}t}{2}\dot{q}, q + \frac{\Delta{}t}{2}\dot{q}\right),
```

the formula of the source divided by ``\Delta{}t``. The division is deliberate: it makes the result
approximate ``L`` itself, and for the midpoint discrete Lagrangian ``L_d`` of a continuous ``L`` it
is that ``L`` exactly.

`nn` is a `NeuralNetwork` of a [`DiscreteLagrangianNetwork`](@ref); with a function in its place,
`Ld(q_0, q_1)` is called directly, which is how the identity above is checked against an analytic
discrete Lagrangian.

The result `f` is called as `f(q, q̇)` on a pair of vectors `q` and `q̇` or on a pair of `n × B`
matrices, and on no other shape: a scalar for the vectors and a `1 × B` array for the matrices, in
the element type of `q`, which ``\Delta{}t`` is converted to. A plain `Ld` is called on one pair of
positions at a time, as two `AbstractVector`s — the columns of the pair on a matrix call — and
must return a number or a one-element array. A `q̇` of another size than `q` throws a
`DimensionMismatch`, and so does a `q` of another row count than ``n`` with the network method; a
timestep that is not positive and finite throws an `ArgumentError`.
"""
function inverse_modified_lagrangian(Ld::Function, Δt::Real)
    Δt = _check_timestep(Δt)
    (q::AbstractArray, q̇::AbstractArray) -> begin
        δ, a, b = _inverse_modified_positions(q, q̇, Δt)
        _plain_values(Ld, a, b, δ)
    end
end

function inverse_modified_lagrangian(
        nn::AbstractNeuralNetworks.NeuralNetwork{<:DiscreteLagrangianArchitecture},
        Δt::Real)
    Δt = _check_timestep(Δt)
    n = dim(nn.architecture) ÷ 2
    (q::AbstractArray, q̇::AbstractArray) -> begin
        size(q, 1) == n ||
            throw(DimensionMismatch("q has $(size(q, 1)) rows, and the architecture needs $n"))
        δ, a, b = _inverse_modified_positions(q, q̇, Δt)
        value = nn(vcat(a, b)) / δ
        q isa AbstractVector && return eltype(q)(only(value))
        convert(Matrix{eltype(q)}, value)
    end
end

# The timestep in the element type of `q`, and the two positions `q ∓ Δt·q̇/2` that the discrete
# Lagrangian is read at: one formation of them for both methods, and the one place that converts
# `Δt`, since a `Float64` timestep must not widen a `Float32` Lagrangian.
function _inverse_modified_positions(q::AbstractArray, q̇::AbstractArray, Δt)
    size(q) == size(q̇) ||
        throw(DimensionMismatch("q and q̇ differ in size: $(size(q)) and $(size(q̇))"))
    δ = eltype(q)(Δt)
    (δ, q - δ * q̇ / 2, q + δ * q̇ / 2)
end

# A plain `Ld` takes one pair of positions, so a batch is one call per column pair: `Ld` receives
# two column vectors and never a matrix. The result is built by `map` and never written into an
# array, so that `Zygote` differentiates it; an empty batch is a `1 × 0` array.
function _plain_values(Ld, a::AbstractMatrix, b::AbstractMatrix, δ)
    T = typeof(δ)
    reshape(Vector{T}(map((x, y) -> T(only(Ld(x, y)) / δ), eachcol(a), eachcol(b))), 1, :)
end

_plain_values(Ld, a::AbstractVector, b::AbstractVector, δ) = typeof(δ)(only(Ld(a, b)) / δ)
