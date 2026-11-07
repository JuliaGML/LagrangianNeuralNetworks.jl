# LagrangianNeuralNetworks.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliagml.github.io/LagrangianNeuralNetworks.jl/stable)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://juliagml.github.io/LagrangianNeuralNetworks.jl/dev)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)
[![Build Status](https://github.com/JuliaGML/LagrangianNeuralNetworks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGML/LagrangianNeuralNetworks.jl/actions/workflows/CI.yml?query=branch%3Amain)

`LagrangianNeuralNetworks.jl` learns a scalar Lagrangian ``L(q, \dot{q})`` with a neural network
and takes the dynamics from it through the Euler–Lagrange equations. It exports the architectures
`LagrangianNeuralNetwork` and `DiscreteLagrangianNetwork`, the supertype `LagrangianArchitecture`,
the default activation `softplus`, the wrapper `NormalisedLagrangian`, which fixes the value and
the momentum of a Lagrangian at one point, and four losses: `LNNLoss`, which compares the
acceleration the learned Lagrangian predicts with accelerations in the data, `EulerLagrangeLoss`,
which asks the learned Lagrangian to satisfy the Euler–Lagrange equations on the same data with no
solve, `VariationalMidpointLoss`, which needs positions only, and `DiscreteEulerLagrangeLoss`,
which trains a discrete Lagrangian on positions through the discrete Euler–Lagrange equations.
`inverse_modified_lagrangian` reads the continuous Lagrangian that a discrete one stands for. The
discrete network's own supertype, `DiscreteLagrangianArchitecture`, is documented and not
exported. `RegularisedLoss` adds weighted penalties to any of them: the regularity penalties
`VelocityHessianRegulariser` and `MixedHessianRegulariser`, and the scale conditions
`ScaleCondition` and `DiscreteScaleCondition`.

The training — the data loader, the batches, the optimiser and `train!` — is
[GeometricMachineLearning.jl](https://github.com/JuliaGNI/GeometricMachineLearning.jl)'s. This
package does not depend on it, so add it yourself, at version 0.9:

```julia
using Pkg
Pkg.add(name = "GeometricMachineLearning", version = "0.9")
```

Then load it by explicit names beside this package:

```julia
using LagrangianNeuralNetworks
using GeometricMachineLearning: Adam, Batch, DataLoader, NeuralNetwork, TrainingOptimizer, train!

arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1)
nn = NeuralNetwork(arch)
input, output = randn(2, 60), randn(1, 60)  # (q, q̇) stacked, and q̈
dl = DataLoader(input, output)
optimizer = TrainingOptimizer(nn.params; algorithm = Adam())
train!(nn, optimizer, dl, Batch(10), 30; loss = LNNLoss(arch))
```

`scripts/reproduction/lnn_pendulum.jl` trains the network on the pendulum.
