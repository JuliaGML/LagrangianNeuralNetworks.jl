# Release Notes

All notable changes to LagrangianNeuralNetworks.jl.

This package is pre-1.0, so *every* minor release is potentially breaking in the sense of
[SemVer](https://semver.org) for `0.x` versions. The sections below name what actually changed,
so that a compat-only bump can be told apart from a rename or a change in results.


## [Unreleased]

### Breaking Changes

* **softplus is the default activation of `LagrangianNeuralNetwork`, in place of `tanh`.** This
  fixes K2: with `tanh`, `LNNLoss` does not learn the 1D harmonic oscillator, and its loss ends near
  1, the loss of the prediction `q̈ = 0`. `softplus(x) = log(1 + exp(x))` is exported. It overflows
  to `Inf` above about 709 in `Float64` and about 88 in `Float32`, with no guard.
  `scripts/reproduction/k2_harmonic_oscillator.jl` trains the oscillator on the seeds 1 to 16 in
  each precision, and exits 1 when fewer than 14 of a precision end below a loss of 0.25. A network
  built with the default activation computes other values than before; pass
  `activation = tanh` for the old network.
* **The field `gradient` of `VariationalMidpointLoss` is now `derivatives`**, a
  `LagrangianDerivatives`. The value of the loss is unchanged.

### New Features

* **`LagrangianArchitecture`, the supertype of a Lagrangian architecture.** `LagrangianNeuralNetwork`
  subtypes it, and `LNNLoss`, `NetworkLoss`, `VariationalMidpointLoss`, `EulerLagrangeLoss` and
  `lagrangian_acceleration` take any subtype that has a `dim` method and a `Chain`.
* **`NormalisedLagrangian(arch; base, value, momentum)`, the normalisation of Offen 2025,
  Prop. 3.1.** It wraps a `LagrangianArchitecture` so that its network is
  `L̃(x) = NN(x) − NN(x_b) − (q̇ − q̇_b)ᵀ(∂_q̇NN(x_b) − p_b) + c_b`, with `L̃(x_b) = value` and
  `∂_q̇L̃(x_b) = momentum` for every value of the parameters. `∂_q̇NN(x_b)` is computed inside the
  forward pass, by a reverse pass over the `Dense` and `Linear` layers of the wrapped chain, so the
  symbolic derivatives and every loss of the package inherit the normalisation. The reverse pass
  supports `tanh`, `softplus` and `identity`; any other activation throws an `ArgumentError` at
  construction. `base`, `value` and `momentum` take the element type of the parameters, in the
  forward pass and in the compiled derivatives. The package now depends on `Random`, for the
  `AbstractRNG` of `initialparameters`, and on `Symbolics`, for `@register_symbolic`.
* **`EulerLagrangeLoss`, the Euler–Lagrange residual loss of Offen 2025, App. B.** It is the mean
  over the samples of `‖∇q̇∇q̇L q̈ + (∇q∇q̇L)ᵀq̇ − ∇qL‖²`, with no solve against the velocity Hessian
  and no normalisation. The one-sample residual is `_euler_lagrange_residual`, unexported.
* **`LagrangianDerivatives(arch)`, with `input_gradient` and `input_hessian`, unexported.** They
  compile the gradient and the Hessian of a scalar network with respect to its input once, for any
  `Architecture` with a `dim` method, and return `2n × B` and `2n × 2n × B` for an input of size
  `2n × B`. With `hessian = false`, only the gradient is compiled, and `input_hessian` throws an
  `ArgumentError`. `lagrangian_acceleration` and `VariationalMidpointLoss` take their derivatives
  from it, with the values of before; `VariationalMidpointLoss` passes `hessian = false`.

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
