# LagrangianNeuralNetworks.jl

`LagrangianNeuralNetworks` learns a scalar Lagrangian ``L(q, \dot{q})`` with a neural network, and
takes the dynamics from that Lagrangian through the Euler–Lagrange equations.

It exports the architectures `LagrangianNeuralNetwork` and
[`DiscreteLagrangianNetwork`](@ref LagrangianNeuralNetworks.DiscreteLagrangianNetwork), the
supertype [`LagrangianArchitecture`](@ref LagrangianNeuralNetworks.LagrangianArchitecture), the
default activation [`softplus`](@ref LagrangianNeuralNetworks.softplus), the wrapper
[`NormalisedLagrangian`](@ref LagrangianNeuralNetworks.NormalisedLagrangian), which fixes the
value and the momentum of a Lagrangian at one point, and four losses:
[`LNNLoss`](@ref LagrangianNeuralNetworks.LNNLoss), which compares the acceleration
that the learned Lagrangian predicts against accelerations in the data,
[`EulerLagrangeLoss`](@ref LagrangianNeuralNetworks.EulerLagrangeLoss), which asks the learned
Lagrangian to satisfy the Euler–Lagrange equations on the same data with no solve,
[`VariationalMidpointLoss`](@ref LagrangianNeuralNetworks.VariationalMidpointLoss), which needs
positions only, and
[`DiscreteEulerLagrangeLoss`](@ref LagrangianNeuralNetworks.DiscreteEulerLagrangeLoss), which
trains a discrete Lagrangian on positions through the discrete Euler–Lagrange equations.
[`inverse_modified_lagrangian`](@ref LagrangianNeuralNetworks.inverse_modified_lagrangian) reads
the continuous Lagrangian that a discrete one stands for. The discrete network's own supertype,
[`DiscreteLagrangianArchitecture`](@ref LagrangianNeuralNetworks.DiscreteLagrangianArchitecture),
is documented and not exported.
[`RegularisedLoss`](@ref LagrangianNeuralNetworks.RegularisedLoss) adds weighted
penalties to any of them: the regularity penalties
[`VelocityHessianRegulariser`](@ref LagrangianNeuralNetworks.VelocityHessianRegulariser) and
[`MixedHessianRegulariser`](@ref LagrangianNeuralNetworks.MixedHessianRegulariser), and the scale
conditions [`ScaleCondition`](@ref LagrangianNeuralNetworks.ScaleCondition) and
[`DiscreteScaleCondition`](@ref LagrangianNeuralNetworks.DiscreteScaleCondition). The training
itself — the data loader, the batches, the optimiser and `train!` —
is [GeometricMachineLearning](https://github.com/JuliaGNI/GeometricMachineLearning.jl)'s.

```julia
using LagrangianNeuralNetworks
using GeometricMachineLearning: NeuralNetwork

arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1)
nn = NeuralNetwork(arch)
loss = LNNLoss(arch)
```

The page [Lagrangian Neural Network](@ref) derives the loss and documents the functions.
