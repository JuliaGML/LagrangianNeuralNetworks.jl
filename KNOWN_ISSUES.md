# Known issues

What is known to be wrong in LagrangianNeuralNetworks and is not fixed.

## Found late

### K1 · With `activation = tanh`, "training with `LNNLoss` reduces the loss" holds on some draws and not on others.

- **location:** `scripts/verification/training_reduces_the_loss.jl`
- **kind:** defect
- **found:** 2026-10-23
- **evidence:**
  The `LNNLoss` training of the script trains for 30 epochs against the accelerations of a second
  network draw and compares the last loss with the first. With `activation = tanh`, run with
  `Random.seed!(seed)` before it, for the seeds 1 to 16, the last loss is not below the first on 2
  of 16 seeds in `Float64` and on none in `Float32`; with 100 and 300 epochs it fails on 3 and on 2
  of 16 in `Float64`, so more epochs do not close it. In `Float32` it also fails at the seed 1234
  (1.0064 against 0.9977). With the default activation, softplus, it holds on all 16 seeds and at
  the seed 1234 in both precisions; 16 draws do not make it a property of the loss. The script
  trains with softplus and requires 14 of the 16 seeds in each precision. The same training, with
  the assertion on one seed, runs in GeometricMachineLearning's suite at the commit this package
  was cut from.

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
