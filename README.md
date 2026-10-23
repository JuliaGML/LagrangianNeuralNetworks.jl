# LagrangianNeuralNetworks.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://juliagml.github.io/LagrangianNeuralNetworks.jl/stable)
[![Latest](https://img.shields.io/badge/docs-latest-blue.svg)](https://juliagml.github.io/LagrangianNeuralNetworks.jl/dev)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)
[![Build Status](https://github.com/JuliaGML/LagrangianNeuralNetworks.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/JuliaGML/LagrangianNeuralNetworks.jl/actions/workflows/CI.yml?query=branch%3Amain)

`LagrangianNeuralNetworks.jl` learns a scalar Lagrangian ``L(q, \dot{q})`` with a neural network
and takes the dynamics from it through the Euler–Lagrange equations. It exports the architecture
`LagrangianNeuralNetwork` and two losses: `LNNLoss`, which compares the acceleration the learned
Lagrangian predicts with accelerations in the data, and `VariationalMidpointLoss`, which needs
positions only.

The training — the data loader, the batches, the optimiser and `train!` — is
[GeometricMachineLearning.jl](https://github.com/JuliaGNI/GeometricMachineLearning.jl)'s. This
package does not depend on it, so add it yourself. Until GeometricMachineLearning 0.9 is
registered, add it from its `main` branch:

```julia
using Pkg
Pkg.add(url = "https://github.com/JuliaGNI/GeometricMachineLearning.jl", rev = "main")
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
