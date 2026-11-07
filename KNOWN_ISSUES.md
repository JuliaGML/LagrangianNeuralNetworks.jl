# Known issues

What is known to be wrong in LagrangianNeuralNetworks and is not fixed.

## Found late

### K1 · "Training with `LNNLoss` reduces the loss" holds on some draws and not on others.

- **location:** `src/loss/lnn_loss.jl`
- **kind:** defect
- **found:** 2026-10-23
- **evidence:**
  The `LNNLoss` training trains against the accelerations of a second network draw. The counts
  below compare the last loss of the training history with the first. At 30 epochs with
  `activation = tanh`, run with `Random.seed!(seed)` before it, for the seeds 1 to 16, the last
  loss is not below the first on 2 of 16 seeds in `Float64` and on none in `Float32`; with 100 and
  300 epochs it fails on 3 and on 2 of 16 in `Float64`, so more epochs do not close it. In
  `Float32` it also fails at the seed 1234 (1.0064 against 0.9977). With the default activation,
  softplus, at 30 epochs in `Float64`, it holds on 16 of 16 seeds when `LNNLoss(arch)` consumes
  the global random number generator between the first network draw and the inputs, and on 13 of
  16 in the draw order of K10 (the seeds 4, 8 and 10 fail). 16 draws do not make it a property of
  the loss. K10 gives the counts at 60 epochs on the whole data set. The same training, with the
  assertion on one seed, runs in GeometricMachineLearning's suite at the commit this package was
  cut from.

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

### K9 · The losses and the penalties do not refuse an empty batch, a column count of the output that differs from the input's, or a timestep that overflows or underflows the data's element type.

- **location:** `src/loss/variational_midpoint_loss.jl`, `src/loss/discrete_euler_lagrange_loss.jl`, `src/architectures/discrete_lagrangian_network.jl`
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
  `DiscreteEulerLagrangeLoss(DiscreteLagrangianNetwork(1; width = 4), 0.1)` has all three: on an
  empty batch, `randn(2, 0)` with an output of `randn(1, 0)`, it gives `NaN`; on an input of 4
  columns and an output of 3 it throws
  `DimensionMismatch: number of columns of each array must match (got (4, 3))`; and with a timestep
  of `1.0e40`, `Inf` in `Float32`, it gives `NaN` on `Float32` data.
  A timestep of `1.0e-46` underflows instead, to `0.0f0` in `Float32`, and no loss refuses it. With
  `Random.seed!(1234)` before the architectures and the data,
  `VariationalMidpointLoss(LagrangianNeuralNetwork(4; width = 4, nhidden = 1), 1.0e-46)` on
  `randn(Float32, 4, 8)` and `randn(Float32, 2, 8)` gives `NaN32`;
  `DiscreteEulerLagrangeLoss(DiscreteLagrangianNetwork(1; width = 4), 1.0e-46)` on
  `randn(Float32, 2, 8)` and `randn(Float32, 1, 8)` gives `1.0259647f0`, the sum of its residual
  `0.025964737f0` at `weight = 0` and of the `1.0f0` that its
  `MixedHessianRegulariser(arch, 1.0e-46)` gives at the zero `Λ`; and
  `inverse_modified_lagrangian(NeuralNetwork(DiscreteLagrangianNetwork(1; width = 4), CPU(),
  Float32), 1.0e-46)` gives `Inf32` on a `Float32` vector of length 1 and `Float32[Inf Inf Inf]` on
  a `1 × 3` pair.

### K10 · A training with `LNNLoss` does not lower the loss on the whole data set on every draw.

- **location:** `src/loss/lnn_loss.jl:121`
- **kind:** defect
- **found:** 2026-11-05
- **evidence:**
  `scripts/verification/training_reduces_the_loss.jl` at commit `3649123`, run in full with
  `julia --startup-file=no --project=scripts scripts/verification/training_reduces_the_loss.jl`,
  trains `LagrangianNeuralNetwork(2; width = 4, nhidden = 1, activation = softplus)` with
  `LNNLoss` and Adam for 60 epochs with `Batch(10)`. The data are 60 points `randn(T, 2, 60)` and
  the accelerations that a first network draw predicts at them; a second draw is trained on them,
  after `Random.seed!(seed)`. The loss on the whole data set at the parameters after the training
  is below its value before the training on 13 of the seeds 1 to 16 in `Float32` (the seeds 7, 9
  and 10 fail) and on 15 of 16 in `Float64` (the seed 10 fails). `LNNLoss` is a relative norm,
  `norm(q̈_NN − q̈) / norm(q̈)`, so its value on the whole data set is not the mean of the values on
  the minibatches that `train!` minimises. The script checks only `VariationalMidpointLoss`.

### K11 · `LNNLoss`'s default constructor takes any value as its acceleration.

- **location:** `src/loss/lnn_loss.jl:101`
- **kind:** defect
- **found:** 2026-11-07
- **evidence:**
  `struct LNNLoss{FT} <: NetworkLoss` has one field and no inner constructor that names its type,
  so `LNNLoss(x)` accepts any `x` as the acceleration; the method
  `LNNLoss(arch::LagrangianArchitecture)` only adds the intended one.
  `LNNLoss(DiscreteLagrangianNetwork(1; width = 4))` is accepted, as a
  `LNNLoss{DiscreteLagrangianNetwork{typeof(softplus)}}`, and fails only when it is called, with
  `MethodError: objects of type DiscreteLagrangianNetwork{typeof(softplus)} are not callable.`,
  rather than an `ArgumentError` at construction.

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
