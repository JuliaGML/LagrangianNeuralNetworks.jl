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

# The split form of `softplus` makes its compiled derivatives the logistic and `σ(1 - σ)`, with no
# tie-break at the origin: both branches have those derivatives there.
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
        @test isapprox(input_gradient(D, x, ps)[1, 1], σ(v); rtol = sqrt(eps(T)))
        @test isapprox(input_hessian(D, x, ps)[1, 1, 1], σ(v) * (1 - σ(v));
            rtol = sqrt(eps(T)))
    end
    # A `max`/`abs` form gives `-1/2` here.
    @test isapprox(input_gradient(D, reshape(T[0], 1, 1), ps)[1, 1], T(0.5);
        rtol = sqrt(eps(T)))
end

# The base form `log(1 + exp(x))` gave `NaN` here in `Float32`, from a preactivation near 47 at the
# velocity 60; the split form keeps the Hessian and the loss finite.
@testset "the Hessian and the loss at a large velocity" begin
    Random.seed!(1234)
    arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1, activation = softplus)
    ps = params(NeuralNetwork(arch, CPU(), Float32))
    D = LagrangianDerivatives(arch)
    for v in Float32[50, 60, 100, 1000, -1000]
        @test all(isfinite, input_hessian(D, reshape(Float32[0.1, v], 2, 1), ps))
    end

    loss = EulerLagrangeLoss(arch)
    values = [loss(ps, reshape(Float32[0.1, v], 2, 1), zeros(Float32, 1, 1))
              for v in Float32[50, 60, 100]]
    @test all(isfinite, values)
end
