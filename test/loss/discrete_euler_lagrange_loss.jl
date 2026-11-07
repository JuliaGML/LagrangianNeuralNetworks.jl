using LagrangianNeuralNetworks
using LagrangianNeuralNetworks: DiscreteLagrangianArchitecture
using AbstractNeuralNetworks: AbstractNeuralNetworks, Architecture, Chain, Dense,
                              IdentityActivation, NetworkLoss, NeuralNetwork, params
using GeometricMachineLearning: CPU, Adam, Batch, DataLoader, NetworkParameters,
                                TrainingOptimizer, train!
using Test
using Zygote
import Random

include("../helpers/eltypes.jl")
include("../helpers/harmonic_oscillator.jl")
include("../helpers/zero_parameters.jl")

Random.seed!(1234)

"""
    hod(T, Δt)

The architecture and the network, of element type `T`, of the oscillator's midpoint discrete
Lagrangian `L_d(a, b) = (b − a)²/(2Δt) − Δt(a + b)²/8`, exactly: one hidden layer of two neurons
with the activation `x²` turns the input `[a; b]` into `((b − a)², (a + b)²)`, and the output layer
weighs them with `1/(2Δt)` and `−Δt/8`. Its inverse modified Lagrangian is `(q̇² − q²)/2`.
"""
function hod(T::DataType, Δt)
    arch = DiscreteLagrangianNetwork(1; width = 2, nhidden = 0, activation = square)
    nn = NeuralNetwork(arch, CPU(), T)
    ps = params(nn)
    ps.L1.W .= [-1 1; 1 1]
    ps.L1.b .= 0
    ps.L2.W .= [1 / (2Δt) -Δt / 8]
    arch, nn
end

# ---------------------------------------------------------------- the architecture

@testset "the architecture of the discrete Lagrangian network ($T)" for T in REAL_ELTYPES
    default = DiscreteLagrangianNetwork(3)
    @test default isa DiscreteLagrangianArchitecture
    @test DiscreteLagrangianArchitecture <: Architecture
    @test !(default isa LagrangianArchitecture)
    @test default.width == 6
    @test default.act === softplus
    @test AbstractNeuralNetworks.dim(default) === 6
    @test length(Chain(default).layers) == default.nhidden + 2

    # `n` of another integer type is stored as an `Int`, and so is the width that defaults to `2n`.
    two = DiscreteLagrangianNetwork(Int32(2); nhidden = 0)
    @test AbstractNeuralNetworks.dim(two) isa Int
    @test AbstractNeuralNetworks.dim(two) === 4
    @test two.width === 4
    @test length(Chain(two).layers) == 2

    wide = DiscreteLagrangianNetwork(1; width = 4, nhidden = 2)
    layers = Chain(wide).layers
    @test length(layers) == wide.nhidden + 2
    @test layers[1] isa Dense{2, 4, true}
    @test layers[2] isa Dense{4, 4, true}
    @test layers[3] isa Dense{4, 4, true}
    @test layers[4] isa Dense{4, 1, false}
    @test layers[4].σ isa IdentityActivation
    for layer in layers[1:3]
        @test layer.σ.σ === softplus
    end

    for (arch, rows) in ((default, 6), (two, 4), (wide, 2))
        @test arch isa Architecture
        @test !(arch isa LagrangianArchitecture)
        output = NeuralNetwork(arch, CPU(), T)(randn(T, rows, 5))
        @test size(output) == (1, 5)
        @test eltype(output) === T
    end
end

@testset "the constructor of the discrete network refuses a bad shape" begin
    for n in (0, -1)
        @test_throws ArgumentError("n must be positive, got $n") DiscreteLagrangianNetwork(n)
    end
    @test_throws ArgumentError("width must be positive, got 0") DiscreteLagrangianNetwork(
        1; width = 0)
    @test_throws ArgumentError("nhidden must be non-negative, got -1") DiscreteLagrangianNetwork(
        1; nhidden = -1)
end

# ---------------------------------------------------------------- the loss against Zygote

@doc raw"""
With no regulariser, the loss must *be* the mean over the samples of the squared discrete
Euler–Lagrange residual of the network itself, `‖D₂L_d(q_{k−1}, q_k) + D₁L_d(q_k, q_{k+1})‖²`.

The residual is assembled here from `Zygote` gradients of `L_d` itself, which is the call the loss
exists in order not to make, one sample at a time. At `n > 1` a mean over the `n·B` entries differs
from the mean over the `B` samples.
"""
function test_residual_against_zygote(T::DataType, n::Integer; Δt = T(0.1), batch_size = 4)
    arch = DiscreteLagrangianNetwork(n; width = 4, nhidden = 1)
    nn = NeuralNetwork(arch, CPU(), T)
    ps = params(nn)
    loss = DiscreteEulerLagrangeLoss(arch, Δt; weight = 0)

    qₖ₋₁ = randn(T, n, batch_size)
    qₖ = randn(T, n, batch_size)
    qₖ₊₁ = randn(T, n, batch_size)
    input = vcat(qₖ₋₁, qₖ)

    Ld(a, b) = sum(nn(vcat(a, b), ps))
    D₂ = reduce(hcat,
        [Zygote.gradient(b -> Ld(qₖ₋₁[:, i], b), qₖ[:, i])[1] for i in 1:batch_size])
    D₁ = reduce(hcat,
        [Zygote.gradient(a -> Ld(a, qₖ₊₁[:, i]), qₖ[:, i])[1] for i in 1:batch_size])

    value = loss(ps, input, qₖ₊₁)
    @test value isa T
    @test value ≈ sum(abs2, D₂ + D₁) / batch_size
    # The four-argument method is the three-argument one, and trailing dimensions are samples.
    @test loss(nn.model, ps, input, qₖ₊₁) == value
    @test loss(ps, reshape(input, 2n, batch_size, 1), reshape(qₖ₊₁, n, batch_size, 1)) ≈
          value
end

@testset "the loss is the discrete Euler–Lagrange residual ($T)" for T in REAL_ELTYPES
    for n in (1, 2, 3)
        test_residual_against_zygote(T, n)
    end
end

# ---------------------------------------------------------------- the oscillator

@doc raw"""
The default regulariser must separate a null discrete Lagrangian from the oscillator's.

The zero-parameter network is `L_d ≡ 0`, so its residual vanishes on any data; its ``\Lambda`` is
zero, so the tamed penalty is 1 at every pair and the loss the weight, and the `:inverse` penalty
`Inf`, which a weight of 0 leaves unevaluated. The oscillator's midpoint ``L_d`` has the residual
``\rho\,q_k`` on the exact data ``q_k = \cos(t + k\Delta{}t)``, with
``\rho = 2(1 - \cos\Delta{}t)/\Delta{}t - \Delta{}t(1 + \cos\Delta{}t)/2``, and the tamed penalty 0,
because ``\|\Lambda^{-1}\|_F^2 = (1 + \Delta{}t^2/4)^{-2} < 10``. The rounding of the data to
`Float32` moves the loss by 0.022 %, so 5 % holds in both precisions.
"""
function test_null_lagrangian_against_the_oscillator(T::DataType; Δt = 0.1)
    t = range(0, 5; length = 50)
    input = T.(vcat(cos.(t)', cos.(t .+ Δt)'))
    output = T.(cos.(t .+ 2Δt)')

    null = DiscreteLagrangianNetwork(1; width = 4)
    ps = zero_parameters!(params(NeuralNetwork(null, CPU(), T)))
    for weight in (1, 0.5)
        @test DiscreteEulerLagrangeLoss(null, Δt; weight)(ps, input, output) === T(weight)
    end
    # A term of weight 0 is not evaluated, so the `Inf` penalty of the null Lagrangian leaves the
    # residual alone instead of making the loss `NaN`.
    inverse = MixedHessianRegulariser(null, Δt; form = :inverse)
    @test inverse(ps, input) === T(Inf)
    unweighted = DiscreteEulerLagrangeLoss(null, Δt; regulariser = inverse, weight = 0)
    @test unweighted(ps, input, output) === zero(T)

    arch, nn = hod(T, T(Δt))
    value = DiscreteEulerLagrangeLoss(arch, Δt)(params(nn), input, output)
    ρ = 2(1 - cos(Δt)) / Δt - Δt * (1 + cos(Δt)) / 2
    @test value isa T
    @test value < 4.0e-8
    @test isapprox(value, ρ^2 * sum(abs2, cos.(t .+ Δt)) / length(t); rtol = 0.05)
end

@testset "the default regulariser separates a null discrete Lagrangian ($T)" for T in REAL_ELTYPES
    test_null_lagrangian_against_the_oscillator(T)
end

# ---------------------------------------------------------------- the gradient and the training

@doc raw"""
The slope of the residual reaches the parameters, and `train!` takes the loss.

The gradient is taken with the `:inverse` form, because the tamed form is 0 wherever
``\|\Lambda^{-1}\|_F^2 \le 10`` and its gradient is then often 0.
"""
function test_loss_gradient_and_training(T::DataType)
    arch = DiscreteLagrangianNetwork(1; width = 4)
    nn = NeuralNetwork(arch, CPU(), T)
    ps = params(nn)
    input, output = randn(T, 2, 30), randn(T, 1, 30)
    loss = DiscreteEulerLagrangeLoss(arch, 0.1;
        regulariser = MixedHessianRegulariser(arch, 0.1; form = :inverse))
    @test typeof(loss) <: NetworkLoss

    gradient = Zygote.gradient(p -> loss(p, input, output), ps)[1]
    @test gradient isa NetworkParameters
    @test keys(gradient) == keys(ps)
    @test eltype(gradient.L1.W) === T
    @test !all(iszero, gradient.L1.W)
    @test all(isfinite, gradient.L1.W)

    dl = DataLoader(input, output; suppress_info = true)
    history = train!(
        nn, TrainingOptimizer(nn.params; algorithm = Adam()), dl, Batch(10), 1;
        loss = DiscreteEulerLagrangeLoss(arch, 0.1), show_progress = false)
    @test eltype(history) === T
    @test all(isfinite, history)
end

@testset "the discrete Euler–Lagrange loss is differentiable and trains ($T)" for T in REAL_ELTYPES
    test_loss_gradient_and_training(T)
end

# ---------------------------------------------------------------- the mixed-Hessian regulariser

@doc raw"""
The penalty must read ``\Lambda = -\Delta{}t\,D_1D_2L_d`` off the input Hessian ``H`` of the
network itself, with ``D_1D_2L_d[i, j] = H[i, n + j]``: the pair `[q_0; q_1]` is the input of a
discrete Lagrangian, and no midpoint is taken.
"""
function test_regulariser_against_zygote(T::DataType; n = 2, Δt = 0.7, batch_size = 3)
    arch = DiscreteLagrangianNetwork(n; width = 4)
    nn = NeuralNetwork(arch, CPU(), T)
    ps = params(nn)
    input = randn(T, 2n, batch_size)
    r = MixedHessianRegulariser(arch, Δt; form = :inverse)

    value = r(ps, input)
    @test value isa T
    expected = sum(1:batch_size) do b
        H = Zygote.hessian(z -> sum(nn(z, ps)), input[:, b])
        sum(abs2, inv(-T(Δt) * H[1:n, (n + 1):(2n)]))
    end / batch_size
    @test value ≈ expected
end

@testset "the regulariser of a discrete network against Zygote ($T)" for T in REAL_ELTYPES
    test_regulariser_against_zygote(T)
end

@doc raw"""
On the oscillator ``\Lambda = \lambda = 1 + \Delta{}t^2/4`` at every pair, so each form has a
closed value; `Δt = 0.7` keeps `Δt²/4 = 0.1225` far above the tolerance and the factor `−Δt` away
from 1. The continuous penalty of the same form on the same pairs has that closed value too.
"""
function test_forms_on_the_oscillator(T::DataType; Δt = 0.7)
    arch, nn = hod(T, T(Δt))
    continuous_arch, continuous_ps = ho(T; m = 1)
    λ = 1 + Δt^2 / 4
    expected = Dict(:inverse => 1 / λ^2,
        :tamed => max(0, 1 - 10λ^2),
        :sigmoid => 1 - 1 / (1 + exp(-0.01λ^2)))
    pairs = randn(T, 2, 5)

    for form in (:inverse, :tamed, :sigmoid)
        value = MixedHessianRegulariser(arch, Δt; form)(params(nn), pairs)
        @test value isa T
        @test isapprox(value, expected[form]; rtol = 64 * eps(T))
        continuous = MixedHessianRegulariser(continuous_arch, Δt; form)(continuous_ps, pairs)
        @test continuous isa T
        @test isapprox(value, continuous; rtol = 64 * eps(T))
    end
end

@testset "the three forms on the oscillator agree with the continuous form ($T)" for T in REAL_ELTYPES
    test_forms_on_the_oscillator(T)
end

@testset "a singular Λ gives the decided values ($T)" for T in REAL_ELTYPES
    for n in (1, 2)
        arch = DiscreteLagrangianNetwork(n; width = 4)
        ps = zero_parameters!(params(NeuralNetwork(arch, CPU(), T)))
        input = randn(T, 2n, 3)
        @test MixedHessianRegulariser(arch, 0.7; form = :inverse)(ps, input) === T(Inf)
        @test MixedHessianRegulariser(arch, 0.7; form = :tamed)(ps, input) === one(T)
        @test MixedHessianRegulariser(arch, 0.7; form = :sigmoid)(ps, input) === T(1 / 2)
    end
end

@testset "the discrete regulariser refuses an unknown form" begin
    arch = DiscreteLagrangianNetwork(1; width = 4)
    @test_throws ArgumentError("form must be :inverse, :tamed or :sigmoid, got :spectral") MixedHessianRegulariser(
        arch, 0.1; form = :spectral)
end

# ---------------------------------------------------------------- the inverse modified Lagrangian

@doc raw"""
`f(q, q̇)` must be the definition itself, `L_d(q − Δt·q̇/2, q + Δt·q̇/2)/Δt`, in the element type of
`q`: a `Float64` timestep must not widen the Lagrangian of a `Float32` network.
"""
function test_inverse_modified_lagrangian_definition(T::DataType; n = 2, Δt = 0.1)
    arch = DiscreteLagrangianNetwork(n; width = 4)
    nn = NeuralNetwork(arch, CPU(), T)
    f = inverse_modified_lagrangian(nn, Δt)
    δ = T(Δt)

    q, q̇ = randn(T, n), randn(T, n)
    value = f(q, q̇)
    @test value isa T
    @test isapprox(value, only(nn(vcat(q - δ * q̇ / 2, q + δ * q̇ / 2))) / δ;
        rtol = 8 * eps(T))

    Q, Q̇ = randn(T, n, 3), randn(T, n, 3)
    matrix = f(Q, Q̇)
    @test size(matrix) == (1, 3)
    @test eltype(matrix) === T
    for b in 1:3
        column = only(nn(vcat(Q[:, b] - δ * Q̇[:, b] / 2, Q[:, b] + δ * Q̇[:, b] / 2))) / δ
        @test isapprox(matrix[1, b], column; rtol = 8 * eps(T))
    end
    # A `Float64` timestep with data of the other precision keeps the data's.
    @test inverse_modified_lagrangian(nn, 0.1)(q, q̇) isa T
end

@testset "the inverse modified Lagrangian is its definition ($T)" for T in REAL_ELTYPES
    test_inverse_modified_lagrangian_definition(T)
end

@doc raw"""
The inverse modified Lagrangian of the midpoint discrete Lagrangian of ``L`` must be ``L`` again,
for a plain function and for a network, with a position–velocity coupling that is not symmetric so
that a swap of the two arguments of ``L_d`` flips a sign.
"""
function test_inverse_modified_lagrangian_identity(T::DataType; n = 2, Δt = 0.1)
    δ = T(Δt)
    L(q, v) = (sum(abs2, v) - sum(abs2, q)) / 2 + T(0.3) * q[1] * v[2]
    Ld(a, b) = δ * L((a + b) / 2, (b - a) / δ)

    f = inverse_modified_lagrangian(Ld, Δt)
    for _ in 1:25
        q, v = 2 .* rand(T, n) .- 1, 2 .* rand(T, n) .- 1
        @test f(q, v) isa T
        @test isapprox(f(q, v), L(q, v); atol = 100 * eps(T))
    end

    _, nn = hod(T, δ)
    g = inverse_modified_lagrangian(nn, Δt)
    for _ in 1:25
        q, v = 2 .* rand(T, 1) .- 1, 2 .* rand(T, 1) .- 1
        @test g(q, v) isa T
        @test isapprox(g(q, v), (v[1]^2 - q[1]^2) / 2; atol = 100 * eps(T))
    end
end

@testset "the midpoint of a Lagrangian gives that Lagrangian back ($T)" for T in REAL_ELTYPES
    test_inverse_modified_lagrangian_identity(T)
end

@doc raw"""
A plain `Ld` is a function of two positions, so its matrix call must be that function on one column
pair at a time: the `b`-th entry of `f(Q, V)` is `Ld(Q[:, b], V[:, b]) / Δt`, never a value that
`Ld` computed from the whole matrix. A `Ld` written for a pair of vectors works whether it returns a
number or a one-element array, and the result is in the element type of `Q` whatever precision `Ld`
or the network returns.
"""
function test_inverse_modified_lagrangian_plain_matrix(T::DataType; n = 2, Δt = 0.1,
        batch_size = 3)
    δ = T(Δt)
    L(q, v) = (sum(abs2, v) - sum(abs2, q)) / 2 + T(0.3) * q[1] * v[2]
    Ld(a, b) = δ * L((a + b) / 2, (b - a) / δ)

    Q = 2 .* rand(T, n, batch_size) .- 1
    V = 2 .* rand(T, n, batch_size) .- 1
    f = inverse_modified_lagrangian(Ld, Δt)

    # One value per column, that of the same pair called alone, and none of them the whole matrix's.
    matrix = f(Q, V)
    @test matrix isa Matrix{T}
    @test size(matrix) == (1, batch_size)
    for b in 1:batch_size
        @test isapprox(matrix[1, b], L(Q[:, b], V[:, b]); atol = 100 * eps(T))
        @test isapprox(matrix[1, b], f(Q[:, b], V[:, b]); rtol = 8 * eps(T))
    end

    # A `Ld` that returns a one-element array per pair, from the reduction itself and from its
    # `vec`. Both are the midpoint form, whose column is `sum(abs2, V[:, b])/2`.
    Ldvec(a, b) = vec(sum(abs2, b - a; dims = 1)) / (2 * δ)
    Ldrow(a, b) = sum(abs2, b - a; dims = 1) / (2 * δ)
    for Ldpair in (Ldvec, Ldrow)
        values = inverse_modified_lagrangian(Ldpair, Δt)(Q, V)
        @test values isa Matrix{T}
        @test size(values) == (1, batch_size)
        for b in 1:batch_size
            @test isapprox(values[1, b], sum(abs2, V[:, b]) / 2; atol = 100 * eps(T))
        end
    end

    # The result is in the element type of `q`, not in that of what `Ld` returns.
    other = T === Float32 ? Float64 : Float32
    Lother(a, b) = other(sum(abs2, b - a)) / (2 * other(δ))
    g = inverse_modified_lagrangian(Lother, Δt)
    @test g(Q[:, 1], V[:, 1]) isa T
    @test g(Q, V) isa Matrix{T}
    @test size(g(Q, V)) == (1, batch_size)

    # A network of the other precision does not widen the result either.
    nn = NeuralNetwork(DiscreteLagrangianNetwork(n; width = 4), CPU(), other)
    h = inverse_modified_lagrangian(nn, Δt)
    @test h(Q[:, 1], V[:, 1]) isa T
    @test h(Q, V) isa Matrix{T}
    @test size(h(Q, V)) == (1, batch_size)

    # `Ld` never sees the matrix: one call per column pair, each in the element type of `q`,
    # although the timestep given here is a `Float64`.
    seen = DataType[]
    recording(a, b) = (push!(seen, eltype(a)); sum(abs2, b - a) / (2 * δ))
    inverse_modified_lagrangian(recording, 0.1)(Q, V)
    @test length(seen) == batch_size
    @test all(==(T), seen)

    # An empty batch is a `1 × 0` array of the same type, not an error.
    empty = f(zeros(T, n, 0), zeros(T, n, 0))
    @test empty isa Matrix{T}
    @test size(empty) == (1, 0)
end

@testset "a plain Ld on vector pairs and on a matrix ($T)" for T in REAL_ELTYPES
    test_inverse_modified_lagrangian_plain_matrix(T)
end

# ---------------------------------------------------------------- the decided edges

# The refusals, each asserted by its exact message. The loss's own timestep guard is reached with a
# regulariser supplied, because the default one is built from the timestep before the body runs; the
# row guard of its input is reached with a weight of 0, because the regulariser checks the rows too.
@testset "the refusals of the discrete loss and of the inverse modified Lagrangian" begin
    arch = DiscreteLagrangianNetwork(1; width = 4)
    nn = NeuralNetwork(arch, CPU(), Float64)
    ps = params(nn)
    r = MixedHessianRegulariser(arch, 0.1; form = :inverse)
    plain = (a, b) -> zero(eltype(a))

    for Δt in (0, -0.1, NaN, Inf)
        message = "the timestep must be positive and finite, got $Δt"
        @test_throws ArgumentError(message) DiscreteEulerLagrangeLoss(arch, Δt; regulariser = r)
        @test_throws ArgumentError(message) MixedHessianRegulariser(arch, Δt)
        @test_throws ArgumentError(message) inverse_modified_lagrangian(nn, Δt)
        @test_throws ArgumentError(message) inverse_modified_lagrangian(plain, Δt)
    end
    for weight in (-1, NaN, Inf)
        message = "a weight must be finite and non-negative, got $weight"
        @test_throws ArgumentError(message) DiscreteEulerLagrangeLoss(
            arch, 0.1; regulariser = r, weight)
    end

    input = randn(3, 4)
    message = "the input has 3 rows, and the architecture needs 2"
    @test_throws DimensionMismatch(message) r(ps, input)
    @test_throws DimensionMismatch(message) DiscreteEulerLagrangeLoss(
        arch, 0.1; weight = 0)(ps, input, randn(1, 4))
    loss = DiscreteEulerLagrangeLoss(arch, 0.1)
    @test_throws DimensionMismatch(message) loss(ps, input, randn(1, 4))
    @test_throws DimensionMismatch("the output has 2 rows, and the architecture needs 1") loss(
        ps, randn(2, 4), randn(2, 4))

    wide = DiscreteLagrangianNetwork(2; width = 4)
    wide_nn = NeuralNetwork(wide, CPU(), Float64)
    f = inverse_modified_lagrangian(wide_nn, 0.1)
    @test_throws DimensionMismatch("q has 3 rows, and the architecture needs 2") f(
        zeros(3), zeros(3))
    @test_throws DimensionMismatch("q and q̇ differ in size: (2,) and (1,)") f(
        zeros(2), zeros(1))
    # The plain-function method has no row count of its own to check, and checks the sizes.
    g = inverse_modified_lagrangian(plain, 0.1)
    @test_throws DimensionMismatch("q and q̇ differ in size: (2,) and (1,)") g(
        zeros(2), zeros(1))
end

function test_nan_data_propagate(T::DataType)
    arch = DiscreteLagrangianNetwork(1; width = 4)
    ps = params(NeuralNetwork(arch, CPU(), T))
    input, output = randn(T, 2, 4), randn(T, 1, 4)
    input[1, 2] = T(NaN)

    value = DiscreteEulerLagrangeLoss(arch, 0.1)(ps, input, output)
    @test value isa T
    @test isnan(value)
end

@testset "NaN data propagate, and a Float64 timestep does not widen Float32 data ($T)" for T in REAL_ELTYPES
    test_nan_data_propagate(T)
    arch = DiscreteLagrangianNetwork(1; width = 4)
    ps = params(NeuralNetwork(arch, CPU(), T))
    @test DiscreteEulerLagrangeLoss(arch, 0.1)(ps, randn(T, 2, 4), randn(T, 1, 4)) isa T
end
