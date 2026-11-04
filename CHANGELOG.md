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
  `LagrangianDerivatives`.
* **`VariationalMidpointLoss` is the mean squared discrete Euler–Lagrange residual plus a weighted
  regulariser, in place of the relative residual.** This fixes K3: the relative residual
  `‖D₂L_d + D₁L_d‖ / ‖[D₂L_d; D₁L_d]‖` is invariant under a scaling of the Lagrangian, is `NaN`
  for a constant one, and no residual excludes a null Lagrangian, whose discrete residual vanishes
  on any data. The value is now the mean over the samples of `‖D₂L_d + D₁L_d‖²`
  (Ober-Blöbaum & Offen 2023, (16)), with no normalisation, plus `weight · regulariser(ps, input)`.
  The constructor is `VariationalMidpointLoss(arch, Δt; regulariser, weight = 1)`, and the default
  regulariser is `MixedHessianRegulariser(arch, Δt; form = :tamed)`, which is 1 for a singular
  discrete Lagrangian: the zero network now gives the loss `weight`, and the true harmonic
  oscillator on exact data at `Δt = 0.1` a loss below `4e-8`. With `weight = 0` the regulariser is
  not evaluated. A timestep that is not positive and finite and a negative, `NaN` or `Inf` weight
  throw an `ArgumentError`, and data of the wrong row count a `DimensionMismatch`. Every value of
  the loss changes.

### New Features

* **`LagrangianArchitecture`, the supertype of a Lagrangian architecture.** `LagrangianNeuralNetwork`
  subtypes it, and `LNNLoss`, `NetworkLoss`, `VariationalMidpointLoss`, `EulerLagrangeLoss` and
  `lagrangian_acceleration` take any subtype that has a `dim` method and a `Chain`.
* **`EulerLagrangeLoss`, the Euler–Lagrange residual loss of Offen 2025, App. B.** It is the mean
  over the samples of `‖∇q̇∇q̇L q̈ + (∇q∇q̇L)ᵀq̇ − ∇qL‖²`, with no solve against the velocity Hessian
  and no normalisation. The one-sample residual is `_euler_lagrange_residual`, unexported.
* **`LagrangianDerivatives(arch)`, with `input_gradient` and `input_hessian`, unexported.** They
  compile the gradient and the Hessian of a scalar network with respect to its input once, for any
  `Architecture` with a `dim` method, and return `2n × B` and `2n × 2n × B` for an input of size
  `2n × B`. With `hessian = false`, only the gradient is compiled, and `input_hessian` throws an
  `ArgumentError`. `lagrangian_acceleration` and `VariationalMidpointLoss` take their derivatives
  from it; `VariationalMidpointLoss` passes `hessian = false` for its residual.
* **The regularity and scale penalties, and `RegularisedLoss`.** `RegularisedLoss(loss, terms...)`
  adds terms `weight => r` to any loss of the package, where a penalty `r` is called as
  `r(ps, input)`; a term of weight 0 is not evaluated.
  * `VelocityHessianRegulariser(arch)`: the mean of `‖(∇q̇∇q̇L)⁻¹‖_F²` (Offen & Ober-Blöbaum 2023),
    squared and in the Frobenius norm.
  * `MixedHessianRegulariser(arch, Δt; form = :tamed)`: a penalty of
    `Λ = −Δt·∂²L_d/∂q₀∂q₁` of the midpoint discrete Lagrangian, which tends to the velocity
    Hessian as `Δt → 0`, in the forms `:inverse` (`‖Λ⁻¹‖_F²`, Offen & Ober-Blöbaum 2023, (11)),
    `:tamed` (`relu(1 − 10/‖Λ⁻¹‖_F²)`, Offen & Ober-Blöbaum 2024, (34)) and `:sigmoid`
    (`1 − sigmoid(0.01·det(Λ)²)`, Lishkova 2023). A singular `Λ` gives `Inf`, `1` and `1/2`.
  * `ScaleCondition(arch; base, value)` and `DiscreteScaleCondition(arch, Δt; base, value)`:
    `(det ∇q̇∇q̇L(x_b) − value)²` and `(det Λ(x_b) − value)²` (Offen 2025, (B.1), (B.2)).

### Tests

* **No test asserts that a training reduces the loss.** The testsets "training with
  VariationalMidpointLoss reduces the loss" and "training with LNNLoss reduces the loss" leave the
  suite, because each asserted a property of one draw (K1).
  `scripts/verification/training_reduces_the_loss.jl` trains with both losses at the settings of
  those testsets, with `LNNLoss` for 60 epochs in place of 30, on the seeds 1 to 16 in each
  precision, and exits 1 when the last loss is below the first on fewer than 14 seeds of a loss
  and a precision.

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
