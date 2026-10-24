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
