using LagrangianNeuralNetworks
using LagrangianNeuralNetworks: LagrangianDerivatives, input_hessian, _mixed_hessian,
                                _regularity_penalty
using GeometricMachineLearning: CPU, Adam, Batch, DataLoader, NetworkParameters,
                                NeuralNetwork,
                                TrainingOptimizer, params, train!
using LinearAlgebra
using Test
using Zygote
import Random

include("../helpers/eltypes.jl")
include("../helpers/harmonic_oscillator.jl")

Random.seed!(1234)

# A random network of `2n` inputs. The shape is the one of the other loss tests, so that `Zygote`
# compiles it once.
function random_network(T, n)
    NeuralNetwork(
        LagrangianNeuralNetwork(2n; width = 4, nhidden = 1, activation = softplus), CPU(), T)
end

# The parameters of `ps` set to zero, so that the network is `L ≡ 0`.
function zero_parameters!(ps)
    for layer in keys(ps), array in values(ps[layer])

        fill!(array, 0)
    end
    ps
end

# ---------------------------------------------------------------- the discrete mixed Hessian

@doc raw"""
`_mixed_hessian(H, Δt)` must be ``-\Delta{}t\,D_1D_2L_d`` of the midpoint discrete Lagrangian, with
``D_1D_2L_d[i, j] = \partial^2L_d/\partial{}a_i\partial{}b_j``.

At `n = 2` the antisymmetric part of the position–velocity block is not zero, so the orientation and
the sign of that term are seen; `Δt = 0.7` keeps `Δt²/4` far above the tolerance.
"""
function test_mixed_hessian(T::DataType; n = 2, Δt = T(0.7))
    nn = random_network(T, n)
    ps = params(nn)
    L(y) = sum(nn(y, ps))
    Ld(a, b) = Δt * L(vcat((a + b) / 2, (b - a) / Δt))

    a, b = randn(T, n), randn(T, n)
    H = Zygote.hessian(L, vcat((a + b) / 2, (b - a) / Δt))
    Λ = _mixed_hessian(H, Δt)
    @test size(Λ) == (n, n)
    @test eltype(Λ) === T

    Hd = Zygote.hessian(z -> Ld(z[1:n], z[(n + 1):(2n)]), vcat(a, b))
    @test Λ ≈ -Δt * Hd[1:n, (n + 1):(2n)]

    if T === Float64
        h = 1.0e-4
        e(i) = (v = zeros(T, n); v[i] = h; v)
        difference = [(Ld(a + e(i), b + e(j)) - Ld(a + e(i), b - e(j)) -
                       Ld(a - e(i), b + e(j)) + Ld(a - e(i), b - e(j))) / (4h^2)
                      for i in 1:n, j in 1:n]
        @test norm(Λ - (-Δt) * difference) / norm(Λ) < 1.0e-5
    end
end

@testset "the mixed Hessian against a finite difference and Zygote ($T)" for T in REAL_ELTYPES
    test_mixed_hessian(T)
end

@testset "the mixed Hessian of the harmonic oscillator ($T)" for T in REAL_ELTYPES
    Δt = T(0.7)
    for m in (1, 0.1)
        arch, ps = ho(T; m)
        H = input_hessian(LagrangianDerivatives(arch), randn(T, 2, 3), ps)
        for b in 1:3
            Λ = _mixed_hessian(H[:, :, b], Δt)
            @test Λ isa Matrix{T}
            @test size(Λ) == (1, 1)
            @test isapprox(only(Λ), m + 0.7^2 / 4; rtol = 16 * eps(T))
            # A timestep of another precision is converted to that of `H`.
            @test _mixed_hessian(H[:, :, b], 0.7) isa Matrix{T}
        end
    end
end

# ---------------------------------------------------------------- the continuous penalty

function test_velocity_hessian_regulariser(T::DataType, n::Integer)
    nn = random_network(T, n)
    ps = params(nn)
    r = VelocityHessianRegulariser(nn.architecture)
    x = randn(T, 2n, 3)

    value = r(ps, x)
    @test value isa T
    expected = sum(1:3) do b
        H = Zygote.hessian(y -> sum(nn(y, ps)), x[:, b])
        sum(abs2, inv(H[(n + 1):(2n), (n + 1):(2n)]))
    end / 3
    @test value ≈ expected
    @test r(ps, reshape(x, 2n, 3, 1)) ≈ value
end

@testset "the velocity-Hessian regulariser against Zygote ($T)" for T in REAL_ELTYPES
    for n in (1, 2)
        test_velocity_hessian_regulariser(T, n)
    end
end

# ---------------------------------------------------------------- the discrete penalties

@doc raw"""
On the oscillator ``\Lambda = \lambda = m + \Delta{}t^2/4`` at every pair, so each form has a closed
value. `m = 0.1` puts the tamed form above its threshold, `m = 1` below it.
"""
function test_forms_on_the_oscillator(T::DataType; Δt = 0.7)
    for m in (1, 0.1)
        arch, ps = ho(T; m)
        λ = m + Δt^2 / 4
        expected = Dict(:inverse => 1 / λ^2,
            :tamed => max(0, 1 - 10λ^2),
            :sigmoid => 1 - 1 / (1 + exp(-0.01λ^2)))
        input = randn(T, 2, 5)
        for form in (:inverse, :tamed, :sigmoid)
            value = MixedHessianRegulariser(arch, Δt; form)(ps, input)
            @test value isa T
            @test isapprox(value, expected[form]; rtol = 64 * eps(T))
        end
    end
end

function test_inverse_form_against_the_chain_rule(T::DataType; n = 2, Δt = T(0.7))
    nn = random_network(T, n)
    ps = params(nn)
    input = randn(T, 2n, 3)
    value = MixedHessianRegulariser(nn.architecture, Δt; form = :inverse)(ps, input)
    @test value isa T
    expected = sum(1:3) do b
        a, c = input[1:n, b], input[(n + 1):(2n), b]
        H = Zygote.hessian(y -> sum(nn(y, ps)), vcat((a + c) / 2, (c - a) / Δt))
        sum(abs2, inv(_mixed_hessian(H, Δt)))
    end / 3
    @test value ≈ expected
    @test MixedHessianRegulariser(nn.architecture, Δt; form = :inverse)(
        ps, reshape(input, 2n, 1, 3)) ≈ value
end

@testset "the three forms of the mixed-Hessian regulariser ($T)" for T in REAL_ELTYPES
    test_forms_on_the_oscillator(T)
    test_inverse_form_against_the_chain_rule(T)
end

@testset "a singular Hessian gives the decided values ($T)" for T in REAL_ELTYPES
    for n in (1, 2)
        nn = random_network(T, n)
        ps = zero_parameters!(params(nn))
        arch = nn.architecture
        input = randn(T, 2n, 3)
        penalties = Dict(form => MixedHessianRegulariser(arch, 0.7; form)(ps, input)
        for form in (:inverse, :tamed, :sigmoid))
        @test penalties[:inverse] === T(Inf)
        @test penalties[:tamed] === one(T)
        @test penalties[:sigmoid] === T(1 / 2)
        @test VelocityHessianRegulariser(arch)(ps, input) === T(Inf)
    end
end

@testset "an unknown form is refused" begin
    arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1, activation = softplus)
    @test_throws ArgumentError("form must be :inverse, :tamed or :sigmoid, got :spectral") MixedHessianRegulariser(
        arch, 0.1; form = :spectral)
end

# ---------------------------------------------------------------- the scale conditions

@testset "the scale conditions on the harmonic oscillator ($T)" for T in REAL_ELTYPES
    arch, ps = ho(T; m = 1)
    for base in (zeros(2), [0.3, -0.2]), input in (randn(T, 2, 4), randn(T, 2, 1, 4))

        continuous = ScaleCondition(arch; base, value = 2.5)(ps, input)
        @test continuous isa T
        @test isapprox(continuous, 2.25; rtol = 64 * eps(T))
        discrete = DiscreteScaleCondition(arch, 0.7; base, value = 1)(ps, input)
        @test discrete isa T
        @test isapprox(discrete, 0.01500625; rtol = 64 * eps(T))
        # A `Float64` value does not widen the result.
        discrete = DiscreteScaleCondition(arch, 0.7; base, value = 0.5)(ps, input)
        @test discrete isa T
        @test isapprox(discrete, 0.38750625; rtol = 64 * eps(T))
    end
end

# ---------------------------------------------------------------- RegularisedLoss

function test_regularised_loss(T::DataType)
    nn = random_network(T, 1)
    ps = params(nn)
    arch = nn.architecture
    bare = EulerLagrangeLoss(arch)
    r₁ = VelocityHessianRegulariser(arch)
    r₂ = ScaleCondition(arch; value = 2)
    input, output = randn(T, 2, 5), randn(T, 1, 5)

    loss = RegularisedLoss(bare, 0.25 => r₁, 3 => r₂)
    value = loss(ps, input, output)
    @test value isa T
    @test value ≈ bare(ps, input, output) + T(0.25) * r₁(ps, input) + 3 * r₂(ps, input)
    @test loss(nn.model, ps, input, output) == value

    unregularised = RegularisedLoss(bare)
    @test unregularised(ps, input, output) isa T
    @test unregularised(ps, input, output) == bare(ps, input, output)

    # The gradient with respect to the parameters reaches the regulariser.
    with_r₁ = RegularisedLoss(bare, 1 => r₁)
    gradient = Zygote.gradient(p -> with_r₁(p, input, output), ps)[1]
    bare_gradient = Zygote.gradient(p -> bare(p, input, output), ps)[1]
    @test gradient isa NetworkParameters
    @test eltype(gradient.L1.W) === T
    @test !isapprox(gradient.L1.W, bare_gradient.L1.W)

    dl = DataLoader(randn(T, 2, 30), randn(T, 1, 30); suppress_info = true)
    history = train!(
        nn, TrainingOptimizer(nn.params; algorithm = Adam()), dl, Batch(10), 1;
        loss = with_r₁, show_progress = false)
    @test eltype(history) === T
    @test all(isfinite, history)
end

@testset "RegularisedLoss adds its weighted terms ($T)" for T in REAL_ELTYPES
    test_regularised_loss(T)
end

# ---------------------------------------------------------------- the decided edges

@testset "the refusals" begin
    arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1, activation = softplus)
    ps = params(NeuralNetwork(arch, CPU(), Float64))
    bare = EulerLagrangeLoss(arch)
    r = VelocityHessianRegulariser(arch)

    for Δt in (0, -0.1, NaN, Inf)
        message = "the timestep must be positive and finite, got $Δt"
        @test_throws ArgumentError(message) MixedHessianRegulariser(arch, Δt)
        @test_throws ArgumentError(message) DiscreteScaleCondition(arch, Δt)
        # With a regulariser of its own, so that the guard of the loss is the one that is reached.
        @test_throws ArgumentError(message) VariationalMidpointLoss(arch, Δt; regulariser = r)
    end
    for value in (0, -1, NaN, Inf)
        message = "the value of the scale condition must be positive and finite, got $value"
        @test_throws ArgumentError(message) ScaleCondition(arch; value)
        @test_throws ArgumentError(message) DiscreteScaleCondition(arch, 0.1; value)
    end
    for weight in (-1, NaN, Inf)
        message = "a weight must be finite and non-negative, got $weight"
        @test_throws ArgumentError(message) RegularisedLoss(bare, weight => r)
        @test_throws ArgumentError(message) VariationalMidpointLoss(
            arch, 0.1; regulariser = r, weight)
    end
    message = "the base point has 3 entries, and the architecture needs 2"
    @test_throws DimensionMismatch(message) ScaleCondition(arch; base = zeros(3))
    @test_throws DimensionMismatch(message) DiscreteScaleCondition(arch, 0.1; base = zeros(3))

    input = randn(3, 4)
    message = "the input has 3 rows, and the architecture needs 2"
    vml = VariationalMidpointLoss(arch, 0.1)
    for functor in (r, MixedHessianRegulariser(arch, 0.1), ScaleCondition(arch),
        DiscreteScaleCondition(arch, 0.1))
        @test_throws DimensionMismatch(message) functor(ps, input)
    end
    @test_throws DimensionMismatch(message) vml(ps, input, randn(1, 4))
    # With weight 0 the regulariser, which checks the rows too, is not evaluated.
    @test_throws DimensionMismatch(message) VariationalMidpointLoss(
        arch, 0.1; regulariser = r, weight = 0)(ps, input, randn(1, 4))
    @test_throws DimensionMismatch("the output has 2 rows, and the architecture needs 1") vml(
        ps, randn(2, 4), randn(2, 4))
end

# At `n = 2` the determinant and the inverse go through LAPACK, which refuses a matrix with a `NaN`.
function test_nan_data_propagate(T::DataType, n::Integer)
    nn = random_network(T, n)
    ps = params(nn)
    arch = nn.architecture
    bare = EulerLagrangeLoss(arch)
    r = VelocityHessianRegulariser(arch)
    input, output = randn(T, 2n, 4), randn(T, n, 4)
    input[1, 2] = T(NaN)

    for form in (:inverse, :tamed, :sigmoid)
        value = MixedHessianRegulariser(arch, 0.1; form)(ps, input)
        @test value isa T
        @test isnan(value)
    end
    for value in (r(ps, input), RegularisedLoss(bare, 0.5 => r)(ps, input, output),
        VariationalMidpointLoss(arch, 0.1)(ps, input, output))
        @test value isa T
        @test isnan(value)
    end
end

@testset "NaN data propagate ($T)" for T in REAL_ELTYPES
    for n in (1, 2)
        test_nan_data_propagate(T, n)
    end
    # A matrix with an `Inf` entry gives `NaN` as well, where LAPACK would throw.
    for form in (:inverse, :tamed, :sigmoid)
        value = _regularity_penalty(T[Inf 1; 1 1], form)
        @test value isa T
        @test isnan(value)
    end
end

@doc raw"""
A term of weight 0 is not evaluated, so a penalty that is `Inf` or `NaN` where the residual is
finite leaves the value finite. The zero network has a singular velocity Hessian, so its
`VelocityHessianRegulariser` is `Inf`.
"""
function test_zero_weight(T::DataType)
    nn = random_network(T, 1)
    ps = zero_parameters!(params(nn))
    arch = nn.architecture
    bare = EulerLagrangeLoss(arch)
    r = VelocityHessianRegulariser(arch)
    input, output = randn(T, 2, 4), randn(T, 1, 4)
    @test r(ps, input) === T(Inf)

    zero_weight = RegularisedLoss(bare, 0 => r)
    @test zero_weight(ps, input, output) isa T
    @test zero_weight(ps, input, output) == bare(ps, input, output)
    unweighted = VariationalMidpointLoss(arch, 0.1; regulariser = r, weight = 0)
    @test unweighted(ps, input, output) === zero(T)
end

@testset "a zero weight is accepted and not evaluated ($T)" for T in REAL_ELTYPES
    test_zero_weight(T)
end
