module LagrangianNeuralNetworks

# The architecture is a `Chain` of `AbstractNeuralNetworks` layers, and its losses are
# `NetworkLoss`es of that package. `Chain` is imported, not reached through `using`, because the
# architecture adds a constructor method to it.
using AbstractNeuralNetworks: AbstractNeuralNetworks, AbstractExplicitLayer, Architecture,
                              Dense, Linear, NetworkLoss, dim
import AbstractNeuralNetworks: Chain
using NeuralNetworkParameters: NetworkParameters
using LinearAlgebra: norm
# The gradient and the Hessian of the network with respect to its input are compiled symbolic
# expressions, not nested `Zygote` calls: see `lagrangian_acceleration`.
import SymbolicNeuralNetworks
using SymbolicNeuralNetworks: SymbolicNeuralNetwork, derivative

include("architectures/lagrangian_neural_network.jl")
include("loss/lnn_loss.jl")
include("loss/variational_midpoint_loss.jl")

export LagrangianNeuralNetwork, LNNLoss, VariationalMidpointLoss

end
