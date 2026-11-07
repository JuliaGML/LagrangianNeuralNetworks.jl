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

The result `f` is called as `f(q, q̇)`: a scalar for vectors `q` and `q̇`, and a `1 × B` array for
`n × B` matrices, in the element type of `q`, which ``\Delta{}t`` is converted to. A plain `Ld` that
returns a single number for a whole matrix, evaluating no column of it by itself, gives that number
in every column. A `q̇` of another size than `q` throws a `DimensionMismatch`, and so does a `q` of
another row count than ``n`` with the network method; a timestep that is not positive and finite
throws an `ArgumentError`.
"""
function inverse_modified_lagrangian(Ld::Function, Δt::Real)
    Δt = _check_timestep(Δt)
    (q, q̇) -> _inverse_modified_value(Ld, q, q̇, Δt)
end

function inverse_modified_lagrangian(
        nn::AbstractNeuralNetworks.NeuralNetwork{<:DiscreteLagrangianArchitecture},
        Δt::Real)
    Δt = _check_timestep(Δt)
    n = dim(nn.architecture) ÷ 2
    (q::AbstractArray, q̇::AbstractArray) -> begin
        size(q, 1) == n ||
            throw(DimensionMismatch("q has $(size(q, 1)) rows, and the architecture needs $n"))
        _inverse_modified_value((a, b) -> nn(vcat(a, b)), q, q̇, Δt)
    end
end

# The value of the inverse modified Lagrangian at the pair `(q, q̇)`: `Ld` at the two positions
# `q ∓ Δt·q̇/2`, divided by `Δt`. A scalar for a vector `q`, a `1 × B` array for an `n × B` one, in
# the element type of `q`, which `Δt` is converted to: a `Float64` timestep must not widen the
# Lagrangian of a `Float32` network.
function _inverse_modified_value(Ld, q::AbstractArray, q̇::AbstractArray, Δt)
    size(q) == size(q̇) ||
        throw(DimensionMismatch("q and q̇ differ in size: $(size(q)) and $(size(q̇))"))
    δ = eltype(q)(Δt)
    value = Ld(q - δ * q̇ / 2, q + δ * q̇ / 2) / δ
    q isa AbstractVector && return only(value)
    # A `Ld` that evaluates the samples together returns the `1 × B` array; one that returns a single
    # number for the whole matrix gives it in every column.
    value isa AbstractArray ? value : fill(value, 1, size(q, 2))
end
