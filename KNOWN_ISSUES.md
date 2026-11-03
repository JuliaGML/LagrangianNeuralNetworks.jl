# Known issues

What is known to be wrong in LagrangianNeuralNetworks and is not fixed.

## Found late

### K1 · "Training with `LNNLoss` reduces the loss" holds on some draws and not on others.

- **location:** `test/architectures/lagrangian_neural_network_tests.jl`
- **kind:** defect
- **found:** 2026-10-23
- **evidence:**
  `test_lnn_training` trains for 30 epochs against the accelerations of a second network draw and
  asserts `loss_array[end] < loss_array[1]`. Run with `Random.seed!(seed)` before the function,
  for the seeds 1 to 16, the assertion fails on 2 of 16 seeds in `Float64` and on none in
  `Float32`; with 100 and 300 epochs it fails on 3 and on 2 of 16 in `Float64`, so more epochs do
  not close it. In `Float32` it also fails at the seed 1234 of the test file (1.0064 against
  0.9977). The testset fixes the seed 1, on which both precisions pass. The assertion is therefore
  a property of that draw, not of the loss. The same function with the same assertion runs in
  GeometricMachineLearning's suite at the commit this package was cut from.

## Found while writing the tutorial

The probes below train on the harmonic oscillator, `L(q, q̇) = (q̇² − q²)/2` in one dimension,
under `Random.seed!(1234)`, with `LagrangianNeuralNetwork(2; width = 4, nhidden = 1)`, Adam with
`linesearch = 1e-2`, `Batch(20)` and 200 epochs, against GeometricMachineLearning 0.9.0. Each
number is one draw. "Acceleration error" is the largest `|q̈_NN + q|` on a 21 × 21 grid of
`[-1, 1]²`; "fit error" is the largest error of the least-squares fit `L ≈ a·NN + b` on that grid.

### K2 · With the default activation `tanh`, `LNNLoss` does not learn the harmonic oscillator.

- **location:** `src/architectures/lagrangian_neural_network.jl:7`
- **kind:** defect
- **found:** 2026-11-03
- **evidence:**
  On 400 uniform samples of `[-1, 1]²` with `q̈ = −q`, the loss goes from 12.9 to 1.01, and the
  acceleration error is 21.7. Width 8 for 100 epochs ends at 1.19 with an acceleration error of
  32.6; `linesearch = 1e-3` ends at 1.66. A loss near 1 is the loss of the prediction `q̈ = 0`. The
  solve against `∇q̇∇q̇NN` in `_euler_lagrange_acceleration` amplifies every sample where that
  Hessian is near zero. With `activation = x -> log(1 + exp(x))` and otherwise the same settings,
  the loss goes from 9.6 to 0.041 and the acceleration error is 0.055.

### K3 · A small `VariationalMidpointLoss` does not imply the learned dynamics.

- **location:** `src/loss/variational_midpoint_loss.jl:106`
- **kind:** defect
- **found:** 2026-11-03
- **evidence:**
  On 400 triples `q(0), q(Δt), q(2Δt)` of `q(t) = A cos(t + φ)`, with `A` uniform in `[0.2, 1]`,
  `φ` uniform in `[0, 2π]` and `Δt = 0.1`, the loss goes from 0.071 to 3.6e-4 with `tanh`, and the
  acceleration error of the trained network is 21.8. The loss divides by `‖[D₂L_d; D₁L_d]‖`, so it
  is scale-invariant, and a Lagrangian that is a total time derivative, `g(q)·q̇`, makes the
  discrete Euler–Lagrange residual vanish for any data. Nothing in the loss excludes it. With
  softplus the loss ends at 2.0e-4 and the acceleration error is 0.17.

### K4 · The learned Lagrangian is not the true one beyond scale and offset, even where the dynamics are learned.

- **location:** —
- **kind:** docs
- **found:** 2026-11-03
- **evidence:**
  In the softplus run of K2 the fit error is 0.53, and in the softplus run of K3 it is 0.51, on a
  grid where `L` itself ranges over `[-0.5, 0.5]`. A Lagrangian is unique at best up to a factor, a
  constant and a total time derivative, and in one dimension inequivalent Lagrangians give the
  same equations of motion. A comparison of the learned and the true Lagrangian is therefore not a
  measure of the result; the manual does not say so.
