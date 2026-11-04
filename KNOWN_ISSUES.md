# Known issues

What is known to be wrong in LagrangianNeuralNetworks and is not fixed.

## Found late

### K1 · With `activation = tanh`, "training with `LNNLoss` reduces the loss" holds on some draws and not on others.

- **location:** `test/architectures/lagrangian_neural_network_tests.jl`
- **kind:** defect
- **found:** 2026-10-23
- **evidence:**
  `test_lnn_training` trains for 30 epochs against the accelerations of a second network draw and
  asserts `loss_array[end] < loss_array[1]`. With `activation = tanh`, run with
  `Random.seed!(seed)` before the function, for the seeds 1 to 16, the assertion fails on 2 of 16
  seeds in `Float64` and on none in `Float32`; with 100 and 300 epochs it fails on 3 and on 2 of 16
  in `Float64`, so more epochs do not close it. In `Float32` it also fails at the seed 1234 of the
  test file (1.0064 against 0.9977). With the default activation, softplus, it holds on all 16
  seeds and at the seed 1234 in both precisions; 16 draws do not make it a property of the loss.
  The testset fixes the seed 1, on which both precisions pass with either activation. The
  assertion is therefore a property of that draw, not of the loss. The same function with the same
  assertion runs in GeometricMachineLearning's suite at the commit this package was cut from.

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

## Found while writing the tutorial

The probes below train on the harmonic oscillator, `L(q, q̇) = (q̇² − q²)/2` in one dimension,
under `Random.seed!(1234)`, with `LagrangianNeuralNetwork(2; width = 4, nhidden = 1)`, a
`TrainingOptimizer` with `algorithm = Adam()` and `linesearch = 1e-2`, `Batch(20)` and 200 epochs,
against GeometricMachineLearning 0.9.0. Each number is one draw. "Acceleration error" is the
largest `|q̈_NN + q|` on a 21 × 21 grid of `[-1, 1]²`; "fit error" is the largest error of the
least-squares fit `L ≈ a·NN + b` on that grid.

### K3 · A small `VariationalMidpointLoss` does not imply the learned dynamics.

- **location:** `src/loss/variational_midpoint_loss.jl:103`
- **kind:** defect
- **found:** 2026-11-03
- **evidence:**
  On 400 triples `q(0), q(Δt), q(2Δt)` of `q(t) = A cos(t + φ)`, with `A` uniform in `[0.2, 1]`,
  `φ` uniform in `[0, 2π]` and `Δt = 0.1`, the loss goes from 0.071 to 3.6e-4 with `tanh`, and the
  acceleration error of the trained network is 21.8. The loss divides by `‖[D₂L_d; D₁L_d]‖`, so it
  is scale-invariant. A Lagrangian that is a total time derivative, `g(q)·q̇`, makes the discrete
  Euler–Lagrange residual vanish for any data when `g` is affine, and leaves a residual of order
  `Δt³` along a smooth trajectory otherwise. Nothing in the loss excludes it. With softplus the
  loss ends at 2.0e-4 and the acceleration error is 0.17.

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
