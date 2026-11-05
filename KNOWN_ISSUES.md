# Known issues

What is known to be wrong in LagrangianNeuralNetworks and is not fixed.

## Found late

### K1 · "Training with `LNNLoss` reduces the loss" holds on some draws and not on others.

- **location:** `scripts/verification/training_reduces_the_loss.jl`
- **kind:** defect
- **found:** 2026-10-23
- **evidence:**
  The `LNNLoss` training trains against the accelerations of a second network draw and compares
  the last loss with the first. At 30 epochs with `activation = tanh`, run with
  `Random.seed!(seed)` before it, for the seeds 1 to 16, the last loss is not below the first on 2
  of 16 seeds in `Float64` and on none in `Float32`; with 100 and 300 epochs it fails on 3 and on 2
  of 16 in `Float64`, so more epochs do not close it. In `Float32` it also fails at the seed 1234
  (1.0064 against 0.9977). With the default activation, softplus, at 30 epochs in `Float64`, it
  holds on 16 of 16 seeds in the draw order of the former test, which drew the inputs after
  `LNNLoss(arch)` had consumed the global random number generator, and on 13 of 16 in the draw
  order of the script (the seeds 4, 8 and 10 fail). 16 draws do not make it a property of the
  loss. The script trains with softplus for 60 epochs and requires 14 of the 16 seeds in each
  precision. The same training, with the assertion on one seed, runs in GeometricMachineLearning's
  suite at the commit this package was cut from.

### K6 · With `activation = identity`, `input_hessian` returns an `Int64` array, not one of the parameters' element type.

- **location:** `src/derivatives.jl:79`
- **kind:** found late
- **found:** 2026-11-04
- **evidence:**
  The network is linear in its input, so the compiled Hessian is the integer literal `0` in every
  entry. For `arch = LagrangianNeuralNetwork(4; activation = identity)`,
  `input_hessian(LagrangianDerivatives(arch), randn(T, 4, 2), NeuralNetwork(arch, CPU(), T).params)`
  has the element type `Int64` and size `(4, 4, 2)` for `T` in `Float32` and `Float64`. The same
  holds for a `NormalisedLagrangian` of that network, whose Hessian is that of the wrapped one.

### K7 · `LNNLoss` does not refuse an input of the wrong row count with a message of its own.

- **location:** `src/loss/lnn_loss.jl`
- **kind:** defect
- **found:** 2026-11-04
- **evidence:**
  `LNNLoss` has no row guard; an input of the wrong row count fails in a `reshape`. A
  `RegularisedLoss` checks no rows itself and passes this error on. With
  `arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1, activation = softplus)`,
  `r = VelocityHessianRegulariser(arch)` and `loss = RegularisedLoss(LNNLoss(arch), 1 => r)`,
  `loss(ps, randn(3, 4), randn(1, 4))` throws
  `DimensionMismatch: new dimensions (1, 4) must be consistent with array length 6`, where
  `EulerLagrangeLoss` and `VariationalMidpointLoss` name the row count that the architecture needs.

### K8 · The input Hessian of a softplus network is not finite in `Float32` at a large preactivation, so the default `VariationalMidpointLoss` can be `NaN`.

- **location:** `src/architectures/lagrangian_neural_network.jl:18`
- **kind:** defect
- **found:** 2026-11-04
- **evidence:**
  The compiled second derivative of `softplus(x) = log(1 + exp(x))` is not finite in `Float32` at
  a large preactivation. For
  `arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1, activation = softplus)`, drawn in
  `Float32` under `Random.seed!(1234)`, and `D = LagrangianDerivatives(arch)`,
  `input_hessian(D, reshape(Float32[0.1, v], 2, 1), ps)` is finite at `v = 10` and `30`, and not
  finite at `v = 100` and `1000`. The default regulariser of `VariationalMidpointLoss` reads this
  Hessian at the midpoint velocity `(q₁ − q₀)/Δt`, which is large for a small `Δt` and positions
  that are not close. With `Random.seed!(1)`, a `Float32`
  draw of the network and `randn(Float32, 2, 10)`, `randn(Float32, 1, 10)` as the data,
  `VariationalMidpointLoss(arch, 0.05)` gives `NaN`, and the same loss with `weight = 0` gives
  `0.018962238`.

### K9 · The losses and the penalties do not refuse an empty batch, a column count of the output that differs from the input's, or a timestep that overflows the data's element type.

- **location:** `src/loss/variational_midpoint_loss.jl`
- **kind:** defect
- **found:** 2026-11-04
- **evidence:**
  With `arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1, activation = softplus)` and
  `Float64` parameters, `VariationalMidpointLoss(arch, 0.1)` on an input of 4 columns and an
  output of 3 throws the broadcast error
  `DimensionMismatch: a has size (1, 4), b has size (1, 3), mismatch at dim 2`. On an empty batch,
  `randn(2, 0)`, `VariationalMidpointLoss`, `VelocityHessianRegulariser` and
  `MixedHessianRegulariser` give `NaN`, where `EulerLagrangeLoss` throws an `ArgumentError`.
  `VariationalMidpointLoss(arch, 1.0e40)` accepts the timestep, which is `Inf` in `Float32`, and
  gives `NaN` on `Float32` data.

## Found while writing the tutorial

The probes below train on the harmonic oscillator, `L(q, q̇) = (q̇² − q²)/2` in one dimension,
under `Random.seed!(1234)`, with `LagrangianNeuralNetwork(2; width = 4, nhidden = 1)`, a
`TrainingOptimizer` with `algorithm = Adam()` and `linesearch = 1e-2`, `Batch(20)` and 200 epochs,
against GeometricMachineLearning 0.9.0. Each number is one draw. "Acceleration error" is the
largest `|q̈_NN + q|` on a 21 × 21 grid of `[-1, 1]²`; "fit error" is the largest error of the
least-squares fit `L ≈ a·NN + b` on that grid.

### K4 · The learned Lagrangian is not the true one beyond scale and offset, even where the dynamics are learned.

- **location:** —
- **kind:** docs
- **found:** 2026-11-03
- **evidence:**
  With softplus, the fit error is 0.53 after `LNNLoss` on 400 uniform samples of `[-1, 1]²` with
  `q̈ = −q`, and 0.51 after `VariationalMidpointLoss` on 400 triples of `q(t) = A cos(t + φ)` with
  `Δt = 0.1`, on a grid where `L` itself ranges over `[-0.5, 0.5]`. A Lagrangian is unique at best
  up to a factor, a constant and a total time derivative, and in one dimension inequivalent
  Lagrangians give the same equations of motion. A comparison of the learned and the true
  Lagrangian is therefore not a measure of the result; the manual does not say so.
