# LagrangianNeuralNetworks.jl

`LagrangianNeuralNetworks` learns a scalar Lagrangian ``L(q, \dot{q})`` with a neural network, and
takes the dynamics from that Lagrangian through the Euler–Lagrange equations.

It exports the architecture `LagrangianNeuralNetwork`, its supertype
[`LagrangianArchitecture`](@ref LagrangianNeuralNetworks.LagrangianArchitecture), its default
activation [`softplus`](@ref LagrangianNeuralNetworks.softplus), the wrapper
[`NormalisedLagrangian`](@ref LagrangianNeuralNetworks.NormalisedLagrangian), which fixes the
value and the momentum of a Lagrangian at one point, and three losses:
[`LNNLoss`](@ref LagrangianNeuralNetworks.LNNLoss), which compares the acceleration
that the learned Lagrangian predicts against accelerations in the data,
[`EulerLagrangeLoss`](@ref LagrangianNeuralNetworks.EulerLagrangeLoss), which asks the learned
Lagrangian to satisfy the Euler–Lagrange equations on the same data with no solve, and
[`VariationalMidpointLoss`](@ref LagrangianNeuralNetworks.VariationalMidpointLoss), which needs
positions only. The training itself — the data loader, the batches, the optimiser and `train!` —
is [GeometricMachineLearning](https://github.com/JuliaGNI/GeometricMachineLearning.jl)'s.

```julia
using LagrangianNeuralNetworks
using GeometricMachineLearning: NeuralNetwork

arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1)
nn = NeuralNetwork(arch)
loss = LNNLoss(arch)
```

The page [Lagrangian Neural Network](@ref) derives the loss and documents the functions.
