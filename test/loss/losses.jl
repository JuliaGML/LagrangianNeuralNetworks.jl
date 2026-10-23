using LagrangianNeuralNetworks
using LagrangianNeuralNetworks: LNNLoss, VariationalMidpointLoss
using GeometricMachineLearning: CPU
using AbstractNeuralNetworks: NeuralNetwork, params
using Test
import Random

include("../helpers/eltypes.jl")

# A loss's `ps` is a `NetworkParameters`, so a bare `NamedTuple` reaches no method of a
# loss of this package rather than a silently different gradient: which of the two AD hands the loss is what an
# untyped argument hides. The four-argument methods of a training loss carry the same annotation,
# and their model slot is reached by `AbstractNeuralNetworks`' own untyped fallback where the wrapper
# is missing.
#
# Not numeric: the element type of the data does not reach the method selection, so this is not run
# over `REAL_ELTYPES`.
@testset "a loss takes a NetworkParameters" begin
    Random.seed!(23)
    lnn_arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1)

    lnn_input, lnn_output = randn(2, 3), randn(1, 3)

    losses = (
        (LNNLoss(lnn_arch), lnn_arch, lnn_input, lnn_output),
        (VariationalMidpointLoss(lnn_arch, 0.05), lnn_arch, lnn_input, lnn_output)
    )
    for (loss, arch, input, output) in losses
        nn = NeuralNetwork(arch, CPU(), Float64)
        ps = params(nn)
        # the wrapper is accepted, and the same number comes back through the `NeuralNetwork`
        # method, which forwards to the wrapper
        @test loss(nn, input, output) == loss(ps, input, output)
        # the bare `NamedTuple` under the wrapper is not
        @test_throws MethodError loss(params(ps), input, output)
    end
end
