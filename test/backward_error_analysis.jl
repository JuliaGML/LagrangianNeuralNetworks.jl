using LagrangianNeuralNetworks
using LagrangianNeuralNetworks: modified_lagrangian, inverse_modified_lagrangian
using AbstractNeuralNetworks: NeuralNetwork
using Test
import Random

include("helpers/eltypes.jl")

Random.seed!(1234)

square(x) = x^2

"""
    oscillator(T)

The network of `L = (q̇² − q²)/2` in one dimension, exactly: the two neurons `q` and `q̇` with the
activation `x²`, and the output weights `(−1/2, 1/2)`.
"""
function oscillator(T::DataType)
    arch = LagrangianNeuralNetwork(2; width = 2, nhidden = 0, activation = square)
    nn = NeuralNetwork(arch, T)
    nn.params.L1.W .= T[1 0; 0 1]
    nn.params.L1.b .= 0
    nn.params.L2.W .= T[-1 // 2 1 // 2]
    nn
end

"""
    gauged(T; γ = 3/10)

The network of `L = (q̇² − q²)/2 + γ q q̇`, exactly: the three neurons `q`, `q̇` and `q + q̇` with
the activation `x²` and the output weights `(−1/2 − γ/2, 1/2 − γ/2, γ/2)`, since
`γ q q̇ = γ((q + q̇)² − q² − q̇²)/2`.
"""
function gauged(T::DataType; γ = T(3 // 10))
    arch = LagrangianNeuralNetwork(2; width = 3, nhidden = 0, activation = square)
    nn = NeuralNetwork(arch, T)
    nn.params.L1.W .= T[1 0; 0 1; 1 1]
    nn.params.L1.b .= 0
    nn.params.L2.W .= T[-1 // 2 - γ / 2 1 // 2 - γ / 2 γ / 2]
    nn
end

# `L_im = (q̇² − b q²)/2`, the exact inverse modified Lagrangian of the midpoint rule on the
# harmonic oscillator, as the network of the two neurons `q` and `q̇`.
function inverse_modified_oscillator(T::DataType, b)
    nn = oscillator(T)
    nn.params.L2.W .= T[-b / 2 1 // 2]
    nn
end

# A `NeuralNetwork` of the same parameters in `Float64`, so that the finite differences are taken in
# the precision the network's own values are compared in.
function same_network(arch, nn, ::Type{T}) where {T}
    wide = NeuralNetwork(arch, T)
    for layer in keys(nn.params), name in keys(nn.params[layer])

        getproperty(getproperty(wide.params, layer), name) .= getproperty(getproperty(nn.params, layer), name)
    end
    wide
end

# ---------------------------------------------------------------- the backward error analysis

@testset "reference: the harmonic oscillator and a gauge term ($T)" for T in REAL_ELTYPES
    δ = T(1 // 10)
    h = Float64(δ)
    tolerance = T === Float64 ? 1.0e-12 : 1.0e-6

    for (nn, plain) in ((oscillator(T), (q, q̇) -> (q̇^2 - q^2) / 2),
        (gauged(T), (q, q̇) -> (q̇^2 - q^2) / 2 + T(3 // 10) * q * q̇))
        L = modified_lagrangian(nn, 1 // 10)
        for q in range(-1, 1; length = 5), q̇ in range(-1, 1; length = 5)

            value = L([q], [q̇])
            @test value isa T
            @test abs(value - plain(q, q̇) - (h^2 / 24) * (q^2 + q̇^2)) ≤ tolerance
        end
    end
end

# The midpoint rule on `L_im = (q̇² − b q²)/2` with `b = ((2/Δt)tan(Δt/2))²` reproduces the exact
# flow of `(q̇² − q²)/2`, so `L_im` is its inverse modified Lagrangian; the `+Δt²/24` direction
# recovers the frequency to `O(Δt⁴)` and the opposite sign only to `O(Δt²)`.
@testset "reference: the sign convention recovers the true frequency ($T)" for T in REAL_ELTYPES
    Δt = 0.1
    b = ((2 / Δt) * tan(Δt / 2))^2
    L = modified_lagrangian(inverse_modified_oscillator(T, b), Δt)
    ω² = -L(T[1], T[0]) / L(T[0], T[1])
    @test ω² isa T
    @test abs(ω² - 1) ≤ 1.0e-5
end

@testset "the discrete network through the chain rule ($T)" for T in REAL_ELTYPES
    arch = DiscreteLagrangianNetwork(1; width = 4)
    nn = NeuralNetwork(arch, T)
    wide = same_network(arch, nn, Float64)
    @test wide.params.L3.W == Float64.(nn.params.L3.W)

    Δt = 1 // 10
    δ = Float64(Δt)
    L = modified_lagrangian(nn, Δt)
    L_im = inverse_modified_lagrangian(wide, Δt)
    # The reference takes central differences of the fourth order, whose truncation error is `O(h⁴)`:
    # a second difference at step `h` has a round-off of `eps·|L|/h²`, and the bracket then divides by
    # `L_q̇q̇`, which is why the step is `1e-3` rather than `1e-4`. `q̇` enters `L_d` as `δ·q̇/2`, so a
    # step of `2e-2` in `q̇` is a step of `1e-3` on the Lagrangian's own inputs.
    hq, hq̇ = 1.0e-3, 2.0e-2
    weights = ((-2, 1.0), (-1, -8.0), (1, 8.0), (2, -1.0))
    second = ((-2, -1.0), (-1, 16.0), (0, -30.0), (1, 16.0), (2, -1.0))
    F(q, q̇) = L_im([q], [q̇])
    slope(f, x, step) = sum(w * f(x + k * step) for (k, w) in weights) / (12 * step)
    curve(f, x, step) = sum(w * f(x + k * step) for (k, w) in second) / (12 * step^2)

    # The second-order formula on the inverse modified Lagrangian, from those differences of it.
    function reference(q, q̇)
        Lq = slope(t -> F(t, q̇), q, hq)
        Lqq = curve(t -> F(t, q̇), q, hq)
        Lq̇q̇ = curve(v -> F(q, v), q̇, hq̇)
        Lqq̇ = sum(w * slope(v -> F(q + k * hq, v), q̇, hq̇) for (k, w) in weights) / (12 * hq)
        F(q, q̇) + δ^2 / 24 * ((Lq - Lqq̇ * q̇)^2 / Lq̇q̇ - Lqq * q̇^2)
    end

    tolerance = T === Float64 ? 1.0e-6 : 1.0e-3
    for (q, q̇) in ((0.4, -0.3), (-0.5, 0.6), (0.2, 0.25))
        value = L(T[q], T[q̇])
        wanted = reference(q, q̇)
        @test value isa T
        @test abs(value - wanted) ≤ tolerance * max(abs(value), abs(wanted))
    end
end

@testset "the edges of the analysis" begin
    wide = NeuralNetwork(LagrangianNeuralNetwork(4; width = 4), Float64)
    @test_throws ArgumentError("the backward error analysis is one-dimensional, and the model has 2 degrees of freedom") modified_lagrangian(
        wide, 0.1)

    nn = oscillator(Float64)
    for Δt in (0, -0.1, NaN, Inf)
        @test_throws ArgumentError("the timestep must be positive and finite, got $Δt") modified_lagrangian(
            nn, Δt)
    end

    # A timestep is used in the parameters' element type, and one that the conversion makes zero or
    # infinite is not the positive, finite timestep that was asked for, in either form of the analysis.
    narrow = NeuralNetwork(DiscreteLagrangianNetwork(1; width = 4), Float32)
    for Δt in (1.0e-46, 1.0e40)
        @test_throws ArgumentError modified_lagrangian(oscillator(Float32), Δt)
        @test_throws ArgumentError modified_lagrangian(narrow, Δt)
    end

    L = modified_lagrangian(nn, 1 // 10)
    @test_throws DimensionMismatch("q has 2 entries, and the analysis is one-dimensional") L(
        [0.1, 0.2], [0.3])
    @test_throws DimensionMismatch("q̇ has 2 entries, and the analysis is one-dimensional") L(
        [0.1], [0.3, 0.4])

    # The output weight of `q̇²` is zero, so `L_q̇q̇` is zero and the bracket is not finite.
    no_velocity = NeuralNetwork(
        LagrangianNeuralNetwork(2; width = 2, nhidden = 0, activation = square), Float64)
    no_velocity.params.L1.W .= [1 0; 0 1]
    no_velocity.params.L1.b .= 0
    no_velocity.params.L2.W .= [-1 // 2 0]
    @test !isfinite(modified_lagrangian(no_velocity, 1 // 10)([0.5], [0.25]))
end
