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

    # `VariationalMidpointLoss` uses the gradient alone, so it compiles no Hessian.
    @test VariationalMidpointLoss(arch, T(0.05)).derivatives.hessian === nothing
end
