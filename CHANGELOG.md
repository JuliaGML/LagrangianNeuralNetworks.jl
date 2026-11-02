# Release Notes

All notable changes to LagrangianNeuralNetworks.jl.

This package is pre-1.0, so *every* minor release is potentially breaking in the sense of
[SemVer](https://semver.org) for `0.x` versions. The sections below name what actually changed,
so that a compat-only bump can be told apart from a rename or a change in results.


## [Unreleased]

### Maintenance

* **Test floors for the merged downgrade check.** The `Downgrade` job resolves `Project.toml` and
  `test/Project.toml` together at their lower bounds and runs the tests there; two floors of
  `test/Project.toml` go up so that it passes.
  * `Zygote = "0.7.5"`, from `"0.7"`. Zygote 0.7.0 to 0.7.4 allow only ForwardDiff 0.10, and
    GeometricMachineLearning 0.6 and later need ForwardDiff 1, so 0.7.5 is the lowest Zygote that
    resolves.
  * `Aqua = "0.8.14"`, from `"0.8"`. On Julia 1.12, the piracy check of Aqua 0.8.13 and earlier
    stops with `FieldError: type Core.TypeName has no field mt`; Aqua 0.8.14 adapts to the method
    tables of Julia 1.12. Aqua 0.8.0 also reports 21 method ambiguities of other packages and fails
    its persistent-task check here.


## [0.1.0] — 2026-11-01

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
* **Install GeometricMachineLearning yourself.** The package does not depend on it. The tests and
  the scripts use the registered GeometricMachineLearning 0.9 (`[compat]` `"0.9"`), with no
  `[sources]` entry for it; the README has the command.
* **Tests in `Float32` and `Float64`.** Every moved testset runs in both precisions and asserts the
  element type of its result. A new test solves the Euler–Lagrange equations of the pendulum
  Lagrangian `L = m q̇²/2 + cos q` from its closed-form derivatives and compares the acceleration
  with `-sin q / m`, at `m = 1` and `m = 2`, to `10 eps(T)`.
* **The manual** has the moved Lagrangian neural network page and links `Batch` to
  ReducedComplexityModeling's manual through its `stable` inventory; GeometricMachineLearning 0.9
  re-exports `Batch` from that package.
* **`[compat] NeuralNetworkParameters = "0.5.3"`.** GeometricMachineLearning 0.9 needs
  GeometricOptimizers 0.9.2, which needs NeuralNetworkParameters 0.5.3 or later, so the test
  environment does not resolve at a lower floor.
* **`scripts/reproduction/lnn_pendulum.jl`** runs under `LNN_SMOKE` in the `Scripts` workflow.
