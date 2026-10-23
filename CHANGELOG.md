# Release Notes

All notable changes to LagrangianNeuralNetworks.jl.

This package is pre-1.0, so *every* minor release is potentially breaking in the sense of
[SemVer](https://semver.org) for `0.x` versions. The sections below name what actually changed,
so that a compat-only bump can be told apart from a rename or a change in results.


## [Unreleased] — targeting 0.1.0

### New Features

* **The package, cut from GeometricMachineLearning.** The Lagrangian neural network,
  `LagrangianNeuralNetwork`, and its two losses, `LNNLoss` and `VariationalMidpointLoss`, move here
  from GeometricMachineLearning at commit `e759f206`, with their history: `git log --follow` of a
  moved file reaches its commits in GeometricMachineLearning. The exports are the three names
  GeometricMachineLearning exports for this code; `lagrangian_acceleration` stays unexported. The
  package depends on AbstractNeuralNetworks, NeuralNetworkParameters and SymbolicNeuralNetworks,
  and not on GeometricMachineLearning: the training — `DataLoader`, `Batch`, `train!` — is
  GeometricMachineLearning's, and the tests and the scripts load it by explicit names.
  Julia 1.12 is the floor, because on Julia 1.11 the gradient of an `LNNLoss` stops with a
  `StackOverflowError` in type inference on macOS and Windows.
* **Tests in `Float32` and `Float64`.** Every moved testset runs in both precisions and asserts the
  element type of its result. A new test solves the Euler–Lagrange equations of the pendulum
  Lagrangian `L = m q̇²/2 + cos q` from its closed-form derivatives and compares the acceleration
  with `-sin q / m`, at `m = 1` and `m = 2`, to `10 eps(T)`.
* **The manual** has the moved Lagrangian neural network page and links to GeometricMachineLearning
  through its `latest` inventory, the one of its `main` branch.
* **`scripts/reproduction/lnn_pendulum.jl`** runs under `LNN_SMOKE` in the `Scripts` workflow.
