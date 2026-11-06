module LagrangianNeuralNetworks

# Each architecture is a `Chain` of `AbstractNeuralNetworks` layers, and the losses are
# `NetworkLoss`es of that package. `Chain` is imported, not reached through `using`, because each
# architecture adds a constructor method to it.
using AbstractNeuralNetworks: AbstractNeuralNetworks, AbstractExplicitLayer, Architecture,
                              Dense, GenericActivation, IdentityActivation, Initializer,
                              Linear, NetworkLoss, NeuralNetworkBackend, TanhActivation,
                              dim,
                              initialparameters
import AbstractNeuralNetworks: Chain
using NeuralNetworkParameters: NetworkParameters, params
using LinearAlgebra: RowMaximum, det, issuccess, lu, norm
using Random: AbstractRNG
# A derivative rule for the determinant of a penalty, which `Zygote` takes.
import ChainRulesCore
using ChainRulesCore: NoTangent
# The gradient and the Hessian of the network with respect to its input are compiled symbolic
# expressions, not nested `Zygote` calls: see `LagrangianDerivatives`.
import SymbolicNeuralNetworks
using SymbolicNeuralNetworks: SymbolicNeuralNetwork, derivative
using Symbolics: Symbolics, @register_derivative, @register_symbolic

"""
    LagrangianArchitecture <: Architecture

The supertype of an architecture whose network is a scalar Lagrangian ``L(q, \\dot{q})`` of an
input that stacks ``q`` on ``\\dot{q}``.

The losses of this package take any `LagrangianArchitecture`. A subtype needs two methods:
`AbstractNeuralNetworks.dim(arch)`, the input dimension ``2n``, and `Chain(arch)`, a chain with a
scalar output. Without a `dim` method, `AbstractNeuralNetworks` logs an error and returns
`nothing`, and the constructor of a loss fails there.
"""
abstract type LagrangianArchitecture <: Architecture end

include("derivatives.jl")
include("architectures/lagrangian_neural_network.jl")
include("architectures/normalised_lagrangian.jl")
include("loss/lnn_loss.jl")
include("loss/euler_lagrange_loss.jl")
include("loss/regularisers.jl")
include("loss/variational_midpoint_loss.jl")

export LagrangianArchitecture, LagrangianNeuralNetwork, NormalisedLagrangian, softplus
export EulerLagrangeLoss, LNNLoss, VariationalMidpointLoss
export RegularisedLoss, VelocityHessianRegulariser, MixedHessianRegulariser
export ScaleCondition, DiscreteScaleCondition

end
