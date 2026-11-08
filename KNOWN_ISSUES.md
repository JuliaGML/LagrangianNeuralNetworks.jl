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

### K9 · The losses and the penalties do not refuse an empty batch or a column count of the output that differs from the input's, and none of them, nor `inverse_modified_lagrangian`, refuses a timestep that overflows or underflows the data's element type.

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
  `randn(Float32, 2, 8)` and `randn(Float32, 1, 8)` gives the sum of its value at `weight = 0` and
  of the `1.0f0` that its `MixedHessianRegulariser(arch, 1.0e-46)` gives at the zero `Λ`; and
  `inverse_modified_lagrangian(NeuralNetwork(DiscreteLagrangianNetwork(1; width = 4), CPU(),
  Float32), 1.0e-46)` gives `±Inf32` on a `Float32` vector of length 1 and
  `Float32[±Inf ±Inf ±Inf]` on a `1 × 3` pair, because the timestep is `0.0f0` there: the value is
  the network's output divided by zero, and the sign of each entry is the draw's.

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

### K13 · A training with `VariationalMidpointLoss` does not learn the harmonic oscillator on every draw.

- **location:** `src/loss/variational_midpoint_loss.jl`
- **kind:** defect
- **found:** 2026-11-08
- **evidence:**
  The data are 400 triples `q(t) = a cos(t + φ)` at `t = 0, Δt, 2Δt` with `Δt = 0.1`, `a` uniform
  in `[0.2, 1]` and `φ` in `[0, 2π)`, drawn after `Random.seed!(1234)` in `Float64`, as the
  tutorial draws them. `LagrangianNeuralNetwork(2; width = 4)` is trained with
  `VariationalMidpointLoss(arch, 0.1)`, its default regulariser, Adam with `linesearch = 1e-2`,
  `Batch(20)` and 200 epochs, against GeometricMachineLearning 0.9.0. The loss draws from the
  global random number generator when it is built, so the order matters. With the loss built once
  before `Random.seed!(seed)`, `acceleration_error` on the `21 × 21` grid of `[-1, 1]²` against
  `q̈ = −q` is 0.039, 0.035 and 0.053 for the seeds 1, 2 and 4, with a last loss of `3.1e-7`,
  `4.1e-7` and `2.8e-6`, and 1.0025 for the seed 3, whose last loss is `3.4e-4`; with
  `regulariser = DiscreteScaleCondition(arch, 0.1; base = zeros(2), value = 1.0)` in place of the
  default, the same four seeds give 0.050, 0.100, 0.081 and 0.103. With the loss built after
  `Random.seed!(seed)`, as the tutorial builds it, the seed 1 gives 0.086 and the seed 3 gives
  1.0005. Four seeds do not give a rate.

## Found in review

### K12 · The docstrings of the four measures cite a section of their source, not an equation number.

- **location:** `src/evaluation.jl`
- **kind:** docs
- **found:** 2026-11-07
- **evidence:**
  `energy`, `acceleration_error` and `trajectory_error` cite "Offen 2025, §7.3", and
  `energy_alignment` cites "Ober-Blöbaum & Offen 2023, §3.1", where the rule of the build asks a
  method for "author, year and equation number". Neither paper is held:
  `ls ~/Research/Library/papers/ | grep -i offen` gives `Offen_2022_SymplecticIntegration…` and
  `Offen_2023_LearningDiscreteLagrangians…`, and `grep -i ober` gives the `OberBlobaum_2013` to
  `OberBlobaum_2021` entries and no 2023 paper.
  The equation number of a measure is therefore not available in this tree, and the section number
  the sources are cited by is what a reader can follow.

## Found by the comparison

### K14 · Under the protocol of the comparison, the `LNNLoss` training of the harmonic oscillator and of the pendulum reaches a loss of `NaN`.

- **location:** `src/loss/lnn_loss.jl:121`
- **kind:** defect
- **found:** 2026-11-08
- **evidence:**
  `scripts/comparison/compare.jl` at commit `f674d1d`, run in full with
  `julia --startup-file=no --project=scripts scripts/comparison/compare.jl`, writes the `lnn` rows
  of `scripts/comparison/results/harmonic_oscillator.csv` and `pendulum.csv` as `failed`, and warns
  "the final loss is NaN" for both. The network is
  `LagrangianNeuralNetwork(2; width = 4, nhidden = 1, activation = softplus)`, trained on the
  21 × 21 grid of the box with Adam at `linesearch = 1e-2`, `Batch(20)` and 200 epochs after
  `Random.seed!(1234)`. The same training, with its history read, has its first non-finite loss at
  epoch 14 on both benchmarks, after a loss of 0.146 on the harmonic oscillator and 0.136 on the
  pendulum. `EulerLagrangeLoss` on a `NormalisedLagrangian` of the same network, on the same data,
  ends at a loss of 3.5e-4 and 1.7e-4. `LNNLoss` solves against the velocity Hessian of the network
  and `EulerLagrangeLoss` does not; whether a singular velocity Hessian in a minibatch gives the
  `NaN` is not checked.

### K15 · The last digits of a `VariationalMidpointLoss` row with `n = 2` differ between runs of the same code.

- **location:** `scripts/comparison/compare.jl`
- **kind:** found late
- **found:** 2026-11-08
- **evidence:**
  Two full runs of `julia --startup-file=no --project=scripts scripts/comparison/compare.jl`, at
  commit `f674d1d` and at `93f0857`, whose code under `src/` and `scripts/` differs only in a
  comment of the script, write the same measures and final loss in every row except the
  `variational_midpoint` rows of the coupled oscillator, Hénon–Heiles and the double pendulum,
  which differ by up to 1e-11 relative. The acceleration error is 0.13290256594040553 and
  0.13290256594040578 on the coupled oscillator, 0.36096109808651955 and 0.36096109808651966 on
  Hénon–Heiles, and 0.9686218642283121 and 0.9686218642320793 on the double pendulum, whose
  trajectory error is 3.143292486981392 and 3.143292486999025. The values agree to the three
  significant digits that `docs/src/comparison.md` prints.
  At `bbc798b`, a process that includes `compare.jl` and calls its `run_row` on the double
  pendulum gives, for the `variational_midpoint` row run alone, the acceleration error
  0.9686218642283121 and the trajectory error 3.143292486981392 in each of two fresh processes:
  the values of the run at `f674d1d`, not those of the committed run at `93f0857`. Run after the
  `lnn` and `euler_lagrange` rows of the same benchmark in one process, the row gives
  0.9686218642284424 and 3.1432924869819274, and the `euler_lagrange` row before it gives the
  acceleration error 0.3212480625822604, where both full runs write 0.32124806258225447. A row's
  last digits therefore depend on what the process ran before it, although every row reseeds its
  random draw. The cause is not located; the order in which the symbolic layer compiles its
  functions is a guess.
