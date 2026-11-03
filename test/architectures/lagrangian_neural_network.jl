using LagrangianNeuralNetworks
using LagrangianNeuralNetworks: lagrangian_acceleration
using AbstractNeuralNetworks: AbstractNeuralNetworks, Architecture, Chain, Dense,
                              GenericActivation, IdentityActivation, NetworkLoss,
                              NeuralNetwork,
                              params
using GeometricMachineLearning: CPU
using Test
import Random

include("../helpers/eltypes.jl")

Random.seed!(1234)

@testset "the default architecture" begin
    arch = LagrangianNeuralNetwork(2)
    @test arch isa LagrangianArchitecture
    @test LagrangianArchitecture <: Architecture
    @test arch.act === softplus

    chain = Chain(arch)
    @test length(chain.layers) == 3
    @test chain.layers[1] isa Dense{2, 2, true}
    @test chain.layers[2] isa Dense{2, 2, true}
    @test chain.layers[3] isa Dense{2, 1, false}
    for layer in chain.layers[1:2]
        @test layer.σ isa GenericActivation
        @test layer.σ.σ === softplus
    end
    @test chain.layers[3].σ isa IdentityActivation
end

@testset "softplus keeps the element type ($T)" for T in REAL_ELTYPES
    @test softplus(T(0)) isa T
    @test softplus(T(0)) == log(T(2))
    for x in (T(1), T(-30), T(20))
        @test softplus(x) isa T
        @test isapprox(softplus(x), log(1 + exp(x)); rtol = 8 * eps(T))
    end
end

# The overflow is documented, not guarded: `exp` overflows above about 709 in `Float64` and about
# 88 in `Float32`.
@testset "softplus overflows to Inf" begin
    @test softplus(1000.0) == Inf
    @test softplus(100.0f0) === Inf32
end

# An architecture that is a `LagrangianArchitecture` only through the abstract type, with `dim` and
# `Chain` as its only methods, as a wrapping architecture has.
struct WrappedLagrangian{AT <: LagrangianNeuralNetwork} <: LagrangianArchitecture
    inner::AT
end

AbstractNeuralNetworks.dim(arch::WrappedLagrangian) = AbstractNeuralNetworks.dim(arch.inner)
AbstractNeuralNetworks.Chain(arch::WrappedLagrangian) = Chain(arch.inner)

@testset "the losses take any LagrangianArchitecture" begin
    inner = LagrangianNeuralNetwork(2; width = 4)
    wrapped = WrappedLagrangian(inner)

    @test NetworkLoss(wrapped) isa LNNLoss
    pairs = (
        (lagrangian_acceleration(wrapped), lagrangian_acceleration(inner)),
        (LNNLoss(wrapped), LNNLoss(inner)),
        (VariationalMidpointLoss(wrapped, 0.05), VariationalMidpointLoss(inner, 0.05)),
        (EulerLagrangeLoss(wrapped), EulerLagrangeLoss(inner))
    )

    for T in REAL_ELTYPES
        ps = params(NeuralNetwork(inner, CPU(), T))
        input, output = randn(T, 2, 5), randn(T, 1, 5)

        a_wrapped, a_inner = pairs[1]
        @test a_wrapped(input, ps) isa Matrix{T}
        @test a_wrapped(input, ps) == a_inner(input, ps)
        for (loss_wrapped, loss_inner) in pairs[2:end]
            @test loss_wrapped(ps, input, output) isa T
            @test loss_wrapped(ps, input, output) == loss_inner(ps, input, output)
        end
    end
end
