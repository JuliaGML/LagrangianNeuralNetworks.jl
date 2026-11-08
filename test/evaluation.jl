using LagrangianNeuralNetworks
using LagrangianNeuralNetworks: energy, acceleration_error, energy_alignment,
                                trajectory_error
using AbstractNeuralNetworks: NeuralNetwork
using LinearAlgebra: I, SingularException, norm
using Test
import Random

include("helpers/eltypes.jl")

Random.seed!(1234)

# The seven names of making a learned model's behaviour measurable, each exported and documented; the
# manual lists them under their qualified names.
@testset "the seven names are exported and documented" begin
    exported = (:DiscreteLagrangian, :variational_rollout, :modified_lagrangian, :energy,
        :acceleration_error, :energy_alignment, :trajectory_error)
    for name in exported
        @test Base.isexported(LagrangianNeuralNetworks, name)
        @test Base.Docs.hasdoc(LagrangianNeuralNetworks, name)
    end
end

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
    coupled(T; γ = 3/10)

The network of `L = ½‖q̇‖² − ½‖q‖² + γ q¹q̇²` in two dimensions, exactly: the five neurons `q¹`,
`q²`, `q̇¹`, `q̇²` and `q¹ + q̇²` with the activation `x²`, and the output weights that give the
squares that combination.
"""
function coupled(T::DataType; γ = T(3 // 10))
    arch = LagrangianNeuralNetwork(4; width = 5, nhidden = 0, activation = square)
    nn = NeuralNetwork(arch, T)
    nn.params.L1.W .= T[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1; 1 0 0 1]
    nn.params.L1.b .= 0
    nn.params.L2.W .= T[(-1 - γ)/2 -1 // 2 1 // 2 (1 - γ)/2 γ/2]
    nn
end

"""
    equivalent(T; ρ = 5/2, c = 7/10)

The network of `ρ(q̇² − q²)/2 + c + q q̇`, which is `ρ(q̇² − q²)/2` plus the constant `c` and the
total derivative of `q²/2`, exactly: the four neurons `q`, `q̇`, `q + q̇` and the constant `√c`, with
the output weights `(−ρ/2 − 1/2, ρ/2 − 1/2, 1/2, 1)`.
"""
function equivalent(T::DataType; ρ = T(5 // 2), c = T(7 // 10))
    arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 0, activation = square)
    nn = NeuralNetwork(arch, T)
    nn.params.L1.W .= T[1 0; 0 1; 1 1; 0 0]
    nn.params.L1.b .= T[0, 0, 0, sqrt(c)]
    nn.params.L2.W .= T[-ρ / 2 - 1 // 2 ρ / 2 - 1 // 2 1 // 2 1]
    nn
end

# `L = −q²/2` alone: the output weight of `q̇²` is zero, so `∇q̇∇q̇L` is singular.
function no_velocity(T::DataType)
    nn = oscillator(T)
    nn.params.L2.W .= T[-1 // 2 0]
    nn
end

# The `2 × 24` states of the `5 × 5` grid of `[−1, 1]²` without the origin, one column each.
function grid_states(T::DataType)
    hcat([T[q, q̇] for q in range(-1, 1; length = 5)
          for q̇ in range(-1, 1; length = 5)
          if !(iszero(q) && iszero(q̇))]...)
end

# Nine states of `[−1, 1]⁴`, none of them the origin.
const COUPLED_STATES = [0.3 -0.2 0.5 0.1 -0.4 0.25 -0.15 0.6 0.45;
                        0.1 0.5 -0.4 0.2 0.35 -0.6 0.5 -0.25 0.15;
                        0.4 -0.6 0.3 0.1 -0.2 0.55 -0.45 0.2 0.35;
                        -0.7 0.25 0.2 -0.3 0.4 -0.15 0.6 -0.5 -0.2]

# ---------------------------------------------------------------- the energy

@testset "the harmonic oscillator's energy ($T)" for T in REAL_ELTYPES
    E = energy(oscillator(T))
    tolerance = T === Float64 ? 1.0e-12 : 1.0e-6
    for q in range(-1, 1; length = 5), q̇ in range(-1, 1; length = 5)

        value = E(T[q], T[q̇])
        @test value isa T
        @test abs(value - (q^2 + q̇^2) / 2) ≤ tolerance
    end
end

@testset "the energy of an equivalent Lagrangian ($T)" for T in REAL_ELTYPES
    ρ, c = 5 // 2, 7 // 10
    E = energy(equivalent(T))
    tolerance = T === Float64 ? 1.0e-12 : 1.0e-5
    for q in range(-1, 1; length = 5), q̇ in range(-1, 1; length = 5)

        value = E(T[q], T[q̇])
        @test value isa T
        @test abs(value - (ρ * (q^2 + q̇^2) / 2 - c)) ≤ tolerance
    end
end

@testset "the energy edges" begin
    @test_throws ArgumentError("energy needs a continuous Lagrangian: a discrete one is a function of two positions, and has no velocity to differentiate") energy(
        NeuralNetwork(DiscreteLagrangianNetwork(1; width = 4), Float64))
    E = energy(oscillator(Float64))
    @test_throws DimensionMismatch("q has 2 entries, and the Lagrangian takes 1") E(
        [0.1, 0.2], [0.3])
    @test_throws DimensionMismatch("q̇ has 2 entries, and the Lagrangian takes 1") E(
        [0.1], [0.3, 0.4])
end

# ---------------------------------------------------------------- the acceleration error

@testset "the coupled network and a scaled reference ($T)" for T in REAL_ELTYPES
    nn = coupled(T)
    γ = T(3 // 10)
    x = T.(COUPLED_STATES)
    reference = vcat(-x[1:1, :] .+ γ .* x[4:4, :], -x[2:2, :] .- γ .* x[3:3, :])

    exact = acceleration_error(nn, x, reference)
    @test exact isa T
    @test exact ≤ (T === Float64 ? 1.0e-12 : 1.0e-5)

    doubled = acceleration_error(nn, x, 2 .* reference)
    @test doubled isa T
    @test isapprox(doubled, T(1 // 2); rtol = T === Float64 ? 1.0e-12 : 1.0e-6)
end

@testset "the acceleration of an equivalent Lagrangian ($T)" for T in REAL_ELTYPES
    x = grid_states(T)
    error = acceleration_error(equivalent(T), x, -x[1:1, :])
    @test error isa T
    @test error ≤ (T === Float64 ? 1.0e-12 : 1.0e-5)
end

@testset "the acceleration edges" begin
    nn = oscillator(Float64)
    x = Float64[0.3 -0.2 0.5; 0.1 0.5 -0.4]
    @test_throws ArgumentError("the reference acceleration is zero, and a relative error against it is not defined") acceleration_error(
        nn, x, zeros(1, 3))
    @test_throws ArgumentError("there are no points to measure the acceleration error at") acceleration_error(
        nn, zeros(2, 0), zeros(1, 0))
    @test_throws DimensionMismatch("x has 3 rows, and the Lagrangian takes 2") acceleration_error(
        nn, zeros(3, 3), zeros(1, 3))
    @test_throws DimensionMismatch("the reference acceleration has 2 rows, and the Lagrangian gives 1") acceleration_error(
        nn, x, zeros(2, 3))
    @test_throws DimensionMismatch("x has 3 columns and the reference acceleration has 2") acceleration_error(
        nn, x, ones(1, 2))
    @test_throws DimensionMismatch("x has 2 columns and the reference acceleration has 3") acceleration_error(
        nn, x[:, 1:2], ones(1, 3))
    @test_throws ArgumentError("acceleration_error needs a continuous Lagrangian: a discrete one is a function of two positions, and has no velocity to differentiate") acceleration_error(
        NeuralNetwork(DiscreteLagrangianNetwork(1; width = 4), Float64), zeros(2, 3), ones(1, 3))
    @test_throws SingularException acceleration_error(no_velocity(Float64), x, ones(1, 3))
end

# ---------------------------------------------------------------- the alignment of the energies

@testset "reference: the determinant in two dimensions ($T)" for T in REAL_ELTYPES
    points = 2 .* rand(T, 2, 20) .- 1
    one = x -> [x[1] + T(3 // 10) * x[2], x[2]]
    two = x -> [2 * x[1], x[2]]

    wanted = [abs(u[1] * v[2] - u[2] * v[1]) / (norm(u) * norm(v))
              for (u, v) in ((one(points[:, b]), two(points[:, b])) for b in 1:20)]
    value = energy_alignment(one, two, points)
    @test value isa T
    @test abs(value - sum(wanted) / 20) ≤ (T === Float64 ? 1.0e-12 : 1.0e-5)
end

@testset "an angle in four dimensions ($T)" for T in REAL_ELTYPES
    points = T[0.5 -0.2 0.7; 0.1 0.3 -0.4; -0.6 0.2 0.5; 0.3 -0.7 0.1]
    value = energy_alignment(x -> T[1, 0, 0, 0], x -> T[3, 0, 4, 0], points)
    @test value isa T
    @test abs(value - T(4 // 5)) ≤ 4 * eps(T)
end

@testset "the gradient of the energy ($T)" for T in REAL_ELTYPES
    x = T.(COUPLED_STATES)
    value = energy_alignment(energy(coupled(T)), state -> state, x)
    @test value isa T
    @test value ≤ (T === Float64 ? 1.0e-12 : 1.0e-5)
end

@testset "invariance under an equivalent Lagrangian ($T)" for T in REAL_ELTYPES
    points = grid_states(T)
    reference = x -> [2 * x[1], x[2]]
    plain = energy_alignment(energy(oscillator(T)), reference, points)
    scaled = energy_alignment(energy(equivalent(T)), reference, points)
    @test plain isa T
    @test scaled isa T
    @test plain > 0
    @test scaled > 0
    @test abs(plain - scaled) ≤ (T === Float64 ? 1.0e-12 : 1.0e-5)
end

@testset "a zero gradient ($T)" for T in REAL_ELTYPES
    E = energy(oscillator(T))
    reference = x -> [2 * x[1], x[2]]
    value = energy_alignment(E, reference, T[0 1; 0 1])
    @test value isa T
    @test abs(value - 1 / sqrt(T(10))) ≤ 4 * eps(T)
    @test_throws ArgumentError("every point has a zero energy gradient, and its alignment is not defined") energy_alignment(
        E, reference, reshape(T[0, 0], 2, 1))
    # Only the reference's gradient is zero at this point, and it is left out as well.
    @test_throws ArgumentError("every point has a zero energy gradient, and its alignment is not defined") energy_alignment(
        E, x -> [x[1], zero(x[1])], reshape(T[0, 1 // 2], 2, 1))
    # Only the energy's gradient is zero at this point, and it is left out as well.
    @test_throws ArgumentError("every point has a zero energy gradient, and its alignment is not defined") energy_alignment(
        x -> zeros(T, 2), x -> T[1, 1], ones(T, 2, 1))
    @test_throws ArgumentError("there are no points to align the energies at") energy_alignment(
        E, reference, zeros(T, 2, 0))
end

# A gradient that is not zero but whose squared norm underflows still has a direction: it makes the
# 45° with `(1, 1)` that a gradient of order one makes.
@testset "a gradient whose squared norm underflows ($T)" for T in REAL_ELTYPES
    value = energy_alignment(x -> T[floatmin(T), 0], x -> T[1, 1], ones(T, 2, 1))
    @test value isa T
    @test abs(value - 1 / sqrt(T(2))) ≤ 4 * eps(T)
end

# A non-zero gradient whose largest entry is 0 and whose other entry is negative still has a
# direction: `(−1, 0)` is at 135° to `(1, 1)`, and its `|sin θ|` is `1/√2`.
@testset "a gradient whose largest entry is zero ($T)" for T in REAL_ELTYPES
    value = energy_alignment(x -> T[-1, 0], x -> T[1, 1], ones(T, 2, 1))
    @test value isa T
    @test abs(value - 1 / sqrt(T(2))) ≤ 8 * eps(T)
    swapped = energy_alignment(x -> T[1, 1], x -> T[-1, 0], ones(T, 2, 1))
    @test swapped isa T
    @test abs(swapped - 1 / sqrt(T(2))) ≤ 8 * eps(T)
end

# A gradient parallel to `(1, 1)` stays parallel at either end of the range: at `floatmax` its norm
# overflows to `Inf`, and at a subnormal the norm is not to scale, so the direction has to be read
# from the largest entry rather than from the norm.
@testset "a parallel gradient at either end of the range ($T)" for T in REAL_ELTYPES
    for s in (floatmax(T), nextfloat(zero(T)))
        value = energy_alignment(x -> s .* T[1, 1], x -> T[1, 1], ones(T, 2, 1))
        @test value isa T
        @test value ≤ 8 * eps(T)
        swapped = energy_alignment(x -> T[1, 1], x -> s .* T[1, 1], ones(T, 2, 1))
        @test swapped isa T
        @test swapped ≤ 8 * eps(T)
    end
end

# The sine is the part of `v̂` orthogonal to `û`, so a `v̂` that a lost direction turns into the zero
# vector reads as parallel. A reference gradient of `s·(1, 1)` against `(1, −1)` is at right angles
# to it, and the sine there is 1 at either end of the range only if the reference is scaled too.
@testset "a perpendicular gradient at either end of the range ($T)" for T in REAL_ELTYPES
    for s in (floatmax(T), nextfloat(zero(T)))
        value = energy_alignment(x -> T[1, -1], x -> s .* T[1, 1], ones(T, 2, 1))
        @test value isa T
        @test abs(value - 1) ≤ 8 * eps(T)
    end
end

# ---------------------------------------------------------------- the trajectory error

@testset "two columns that differ ($T)" for T in REAL_ELTYPES
    difference = T[3 0 0; 4 1 0]
    @test trajectory_error(difference, zeros(T, 2, 3)) === T(5)
    @test trajectory_error(zeros(T, 2, 3), -difference) === T(5)
end

@testset "the trajectory edges ($T)" for T in REAL_ELTYPES
    @test_throws DimensionMismatch("the trajectories differ in size: (2, 3) and (2, 4)") trajectory_error(
        zeros(T, 2, 3), zeros(T, 2, 4))
    @test_throws ArgumentError("there are no columns to compare") trajectory_error(
        zeros(T, 2, 0), zeros(T, 2, 0))
    @test isnan(trajectory_error(T[NaN 0; 0 0], zeros(T, 2, 2)))
end
