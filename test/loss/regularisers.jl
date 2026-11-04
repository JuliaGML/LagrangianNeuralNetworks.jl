using LagrangianNeuralNetworks
using LagrangianNeuralNetworks: LagrangianDerivatives, input_hessian, _det, _mixed_hessian,
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
include("../helpers/zero_parameters.jl")

Random.seed!(1234)

# A random network of `2n` inputs. The shape is the one of the other loss tests, so that `Zygote`
# compiles it once.
function random_network(T, n)
    NeuralNetwork(
        LagrangianNeuralNetwork(2n; width = 4, nhidden = 1, activation = softplus), CPU(), T)
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

# Whether the parameter gradient of the penalty `r` returns, with every leaf finite and of
# element type `T`.
function finite_gradient(r, ps, input, T)
    gradient = Zygote.gradient(p -> r(p, input), ps)[1]
    leaves = [array for layer in keys(gradient) for array in values(gradient[layer])]
    all(leaf -> eltype(leaf) === T && all(isfinite, leaf), leaves)
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
        # The determinant's derivative does not invert the singular matrix.
        @test finite_gradient(MixedHessianRegulariser(arch, 0.7; form = :sigmoid), ps, input, T)
    end
end

@doc raw"""
The derivative of `_det` is the cofactor matrix, which `det`'s own derivative, `det(A)·inv(A)ᵀ`,
equals where `A` is regular; `randn` gives a matrix that is not symmetric for `n ≥ 2`, so the
transpose (the adjugate) is seen. At a singular matrix the cofactors are 1 for `n = 1` and 0 for the
`2 × 2` zero matrix. The factor 3 and the square make the seed of the pullback other than 1.
"""
function test_determinant_derivative(T::DataType)
    for n in (1, 2, 3)
        A = randn(T, n, n)
        gradient = Zygote.gradient(A -> _det(A)^2, A)[1]
        @test eltype(gradient) === T
        @test gradient ≈ Zygote.gradient(A -> det(A)^2, A)[1]
    end
    @test Zygote.gradient(A -> 3 * _det(A), zeros(T, 1, 1))[1] == fill(T(3), 1, 1)
    @test Zygote.gradient(A -> 3 * _det(A), zeros(T, 2, 2))[1] == zeros(T, 2, 2)
end

@testset "the derivative of the determinant ($T)" for T in REAL_ELTYPES
    test_determinant_derivative(T)
end

@testset "an unknown form is refused" begin
    arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1, activation = softplus)
    @test_throws ArgumentError("form must be :inverse, :tamed or :sigmoid, got :spectral") MixedHessianRegulariser(
        arch, 0.1; form = :spectral)
    @test_throws ArgumentError("form must be :inverse, :tamed or :sigmoid, got \"tamed\"") MixedHessianRegulariser(
        arch, 0.1; form = "tamed")
end

@doc raw"""
A matrix whose determinant underflows to 0 is not singular: `s·I` of size 3 has the determinant
``s^3``, which is 0 in floating point for these `s`, and ``\|(sI)^{-1}\|_F^2 = 3/s^2`` is finite.
"""
function test_underflowing_determinant(T::DataType)
    s = T === Float32 ? 1.0f-16 : 1.0e-110
    Λ = Matrix{T}(s * I, 3, 3)
    @test iszero(det(Λ))
    value = _regularity_penalty(Λ, :inverse)
    @test value isa T
    @test isapprox(value, 3 / s^2; rtol = 8 * eps(T))
end

@testset "a determinant that underflows is no singular matrix ($T)" for T in REAL_ELTYPES
    test_underflowing_determinant(T)
end

# ---------------------------------------------------------------- the scale conditions

@testset "the scale conditions on the harmonic oscillator ($T)" for T in REAL_ELTYPES
    # On the zero network the Hessians are singular, and the parameter gradient still returns.
    for n in (1, 2)
        nn = random_network(T, n)
        ps = zero_parameters!(params(nn))
        arch = nn.architecture
        input = randn(T, 2n, 3)
        @test finite_gradient(ScaleCondition(arch; value = 2.5), ps, input, T)
        @test finite_gradient(DiscreteScaleCondition(arch, 0.7; value = 2.5), ps, input, T)
    end

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

@doc raw"""
The oscillator's Hessian is the same at every point, so only a network whose Hessian varies shows
that the conditions read `base`: the continuous one as the point ``[q; \dot{q}]``, the discrete one
as the pair ``[q_0; q_1]``, against `Zygote` Hessians of ``L`` and of ``L_d``.
"""
function test_scale_conditions_at_a_base(T::DataType, n::Integer; Δt = 0.7)
    nn = random_network(T, n)
    ps = params(nn)
    L(y) = sum(nn(y, ps))
    Ld(z) = T(Δt) *
            L(vcat((z[1:n] + z[(n + 1):(2n)]) / 2, (z[(n + 1):(2n)] - z[1:n]) / T(Δt)))
    base = randn(2n)
    input = randn(T, 2n, 3)

    H = Zygote.hessian(L, T.(base))
    expected = (det(H[(n + 1):(2n), (n + 1):(2n)]) - T(2.5))^2
    value = ScaleCondition(nn.architecture; base, value = 2.5)(ps, input)
    @test value isa T
    @test value ≈ expected

    Hd = Zygote.hessian(Ld, T.(base))
    expected = (det(-T(Δt) * Hd[1:n, (n + 1):(2n)]) - T(2.5))^2
    value = DiscreteScaleCondition(nn.architecture, Δt; base, value = 2.5)(ps, input)
    @test value isa T
    @test value ≈ expected
end

@testset "the scale conditions at a base point of a random network ($T)" for T in REAL_ELTYPES
    for n in (1, 2)
        test_scale_conditions_at_a_base(T, n)
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
    # `RegularisedLoss` checks no rows itself: its loss is evaluated first and refuses the input.
    @test_throws DimensionMismatch("EulerLagrangeLoss: the input has 3 rows, and the network takes 2") RegularisedLoss(
        bare, 1 => r)(ps, input, randn(1, 4))
end

# At `n = 2` the inverse goes through LAPACK, which refuses a matrix with a `NaN`.
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
    # A matrix with an `Inf` entry gives `NaN` as well: `det` returns `Inf` there, and `inv` throws.
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
