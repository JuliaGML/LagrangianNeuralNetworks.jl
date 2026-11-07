using LagrangianNeuralNetworks
using LagrangianNeuralNetworks: LagrangianDerivatives, input_gradient, input_hessian
using AbstractNeuralNetworks: AbstractNeuralNetworks, Architecture, Chain, Dense, Linear,
                              NeuralNetwork, params
using GeometricMachineLearning: CPU
using Test
using Zygote
import Random

include("helpers/eltypes.jl")

Random.seed!(1234)

"""
Compare the compiled derivatives of `arch` with `Zygote`'s, sample by sample, on a `2n × 3` input.
The samples differ, so a Hessian or a gradient taken from the wrong sample fails.
"""
function test_derivatives_against_zygote(arch, D, T)
    nn = NeuralNetwork(arch, CPU(), T)
    ps = params(nn)
    d = AbstractNeuralNetworks.dim(arch)
    x = randn(T, d, 3)

    g = input_gradient(D, x, ps)
    H = input_hessian(D, x, ps)
    @test size(g) == (d, 3)
    @test size(H) == (d, d, 3)
    @test eltype(g) === T
    @test eltype(H) === T

    for b in axes(x, 2)
        xb = x[:, b]
        g_ref = Zygote.gradient(y -> sum(nn(y, ps)), xb)[1]
        H_ref = Zygote.hessian(y -> sum(nn(y, ps)), xb)
        @test isapprox(g[:, b], g_ref; rtol = sqrt(eps(T)))
        @test isapprox(H[:, :, b], H_ref; rtol = sqrt(eps(T)))
    end
end

@testset "the derivatives per sample ($T, n = $n)" for T in REAL_ELTYPES, n in (1, 2)

    arch = LagrangianNeuralNetwork(2n; width = 4)
    test_derivatives_against_zygote(arch, LagrangianDerivatives(arch), T)
end

# A network with a scalar output that is no `LagrangianArchitecture`, as a discrete Lagrangian
# network is.
struct ScalarNetwork <: Architecture end

AbstractNeuralNetworks.dim(::ScalarNetwork) = 4
AbstractNeuralNetworks.Chain(::ScalarNetwork) = Chain(Dense(4, 4, softplus), Linear(4, 1))

@testset "the derivatives of a plain architecture ($T)" for T in REAL_ELTYPES
    arch = ScalarNetwork()
    @test !(arch isa LagrangianArchitecture)
    test_derivatives_against_zygote(arch, LagrangianDerivatives(arch), T)
end

@testset "trailing dimensions and a wrong height ($T)" for T in REAL_ELTYPES
    arch = LagrangianNeuralNetwork(4; width = 4)
    D = LagrangianDerivatives(arch)
    ps = params(NeuralNetwork(arch, CPU(), T))

    x = randn(T, 4, 3, 5)
    g = input_gradient(D, x, ps)
    H = input_hessian(D, x, ps)
    @test size(g) == (4, 3, 5)
    @test size(H) == (4, 4, 3, 5)
    @test eltype(g) === T
    @test eltype(H) === T
    @test g == reshape(input_gradient(D, reshape(x, 4, :), ps), 4, 3, 5)
    @test H == reshape(input_hessian(D, reshape(x, 4, :), ps), 4, 4, 3, 5)

    for f in (input_gradient, input_hessian)
        err = try
            f(D, randn(T, 3, 5), ps)
        catch e
            e
        end
        @test err isa DimensionMismatch
        @test occursin(r"^the input has 3 rows, and the network takes 4$", err.msg)
    end
end

@testset "the derivatives without the Hessian ($T)" for T in REAL_ELTYPES
    arch = LagrangianNeuralNetwork(4; width = 4)
    D = LagrangianDerivatives(arch)
    D0 = LagrangianDerivatives(arch; hessian = false)
    @test D0.hessian === nothing
    ps = params(NeuralNetwork(arch, CPU(), T))
    x = randn(T, 4, 3)

    g0 = input_gradient(D0, x, ps)
    @test eltype(g0) === T
    @test isapprox(g0, input_gradient(D, x, ps); rtol = 8 * eps(T))

    err = try
        input_hessian(D0, x, ps)
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin(
        r"^these derivatives were compiled with `hessian = false`, and hold no input Hessian$",
        err.msg)

    # The residual of `VariationalMidpointLoss` uses the gradient alone, so its own derivatives
    # hold no Hessian; its regulariser compiles its own.
    @test VariationalMidpointLoss(arch, T(0.05)).derivatives.hessian === nothing
end

# A network whose value is `softplus` of its one input exactly: one neuron of weight 1 and bias 0,
# then a weight-1 linear layer.
struct UnitNetwork <: Architecture end

AbstractNeuralNetworks.dim(::UnitNetwork) = 1
function AbstractNeuralNetworks.Chain(::UnitNetwork)
    Chain(Dense(1, 1, softplus), Linear(1, 1; use_bias = false))
end

# The registered derivative of `softplus` is the logistic, and that of the logistic `σ(1 - σ)`, so
# the compiled derivatives are those two functions and no branch of either definition is
# differentiated: the tie-break at the origin changes nothing.
@testset "the compiled derivatives of softplus against the logistic ($T)" for T in REAL_ELTYPES
    arch = UnitNetwork()
    ps = params(NeuralNetwork(arch, CPU(), T))
    ps.L1.W .= 1
    ps.L1.b .= 0
    ps.L2.W .= 1
    D = LagrangianDerivatives(arch)

    σ(x) = 1 / (1 + exp(-x))
    for v in (T(-3), T(-1.0e-6), T(0), T(1.0e-6), T(3))
        x = reshape([v], 1, 1)
        g = input_gradient(D, x, ps)[1, 1]
        H = input_hessian(D, x, ps)[1, 1, 1]
        @test g isa T
        @test H isa T
        @test isapprox(g, σ(v); rtol = sqrt(eps(T)))
        @test isapprox(H, σ(v) * (1 - σ(v)); rtol = sqrt(eps(T)))
    end
end

# The registered derivatives keep the Hessian and the loss finite at velocities where `exp` of the
# first layer's preactivation overflows, up to `|v| = 1000`.
@testset "the Hessian and the loss at a large velocity ($T)" for T in REAL_ELTYPES
    Random.seed!(1234)
    arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1, activation = softplus)
    ps = params(NeuralNetwork(arch, CPU(), T))
    D = LagrangianDerivatives(arch)
    for v in T[50, 60, 100, 1000, -1000]
        H = input_hessian(D, reshape(T[0.1, v], 2, 1), ps)
        @test eltype(H) === T
        @test all(isfinite, H)
    end

    loss = EulerLagrangeLoss(arch)
    values = [loss(ps, reshape(T[0.1, v], 2, 1), zeros(T, 1, 1)) for v in T[50, 60, 100]]
    @test all(value -> value isa T, values)
    @test all(isfinite, values)
    # The `Float32` draw is the measured one, `0.19830064`; a finite result that is not it is a
    # different mistake from the overflow.
    if T === Float32
        @test isapprox(values[1], 0.19830064f0; rtol = 1.0e-5)
    end
end

# The parameter gradient runs back through the compiled expression, and so through the logistic,
# whose two branches keep `exp` from overflowing: a single branch gives `Inf / Inf` or `inv(Inf)`
# times `-Inf`, a `NaN` that the value alone can miss.
@testset "the parameter gradient at a large velocity ($T)" for T in REAL_ELTYPES
    Random.seed!(1234)
    arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1, activation = softplus)
    ps = params(NeuralNetwork(arch, CPU(), T))
    loss = EulerLagrangeLoss(arch)

    for v in T[0.5, 50, 100, 1000, -1000]
        input, output = reshape(T[0.1, v], 2, 1), zeros(T, 1, 1)
        value = loss(ps, input, output)
        @test value isa T
        @test isfinite(value)
        dp = Zygote.gradient(p -> loss(p, input, output), ps)[1]
        @test eltype(dp.L1.W) === T
        for layer in (:L1, :L2)
            @test all(isfinite, getproperty(dp, layer).W)
            @test all(isfinite, getproperty(dp, layer).b)
        end
        @test all(isfinite, dp.L3.W)
    end
end
