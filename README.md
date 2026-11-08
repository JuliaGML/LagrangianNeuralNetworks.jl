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
package does not depend on it, so install the two together:

```julia
using Pkg
Pkg.add(["LagrangianNeuralNetworks", "GeometricMachineLearning"])
```

Then load GeometricMachineLearning by explicit names beside this package. The example learns the
harmonic oscillator, `L = (q̇² − q²)/2`, from its acceleration `q̈ = −q` on a grid of states, and
prints the relative error of the acceleration that the learned Lagrangian predicts there:

```julia
using LagrangianNeuralNetworks
using GeometricMachineLearning: CPU, Adam, Batch, DataLoader, NeuralNetwork, TrainingOptimizer,
                                train!

# The states [q; q̇] on a 21 × 21 grid of [-1, 1]², and the acceleration q̈ = −q at each.
input = hcat([[q, q̇] for q in -1:0.1:1 for q̇ in -1:0.1:1]...)
output = -input[1:1, :]

arch = LagrangianNeuralNetwork(2; width = 4)
nn = NeuralNetwork(arch, CPU(), Float64)
dl = DataLoader(input, output; suppress_info = true)
optimizer = TrainingOptimizer(nn.params; algorithm = Adam(), linesearch = 1e-2)
train!(nn, optimizer, dl, Batch(21), 200; loss = LNNLoss(arch), show_progress = false)

println("acceleration error: ", acceleration_error(nn, input, output))
```

The [tutorial](https://juliagml.github.io/LagrangianNeuralNetworks.jl/dev/tutorials/lagrangian_neural_network/)
trains four models of the oscillator, two on accelerations and two on positions only, and compares
them through measures that do not depend on which of the many equivalent Lagrangians a network
learns. The
[architecture page](https://juliagml.github.io/LagrangianNeuralNetworks.jl/dev/architectures/lagrangian_neural_network/)
describes each method with its source. `scripts/reproduction/lnn_pendulum.jl` trains the network
on the pendulum.
