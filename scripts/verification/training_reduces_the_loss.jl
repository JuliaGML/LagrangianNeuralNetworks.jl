# Train a Lagrangian neural network with each of two losses, on the seeds 1 to 16 in each precision,
# and count the seeds on which the last loss of the history is below the first.
#
# `VariationalMidpointLoss`, with its default regulariser and weight: the triples
# `(q(t), q(t + Δt), q(t + 2Δt))` of `q(t) = cos t` at `Δt = 0.05`, 60 of them, `Batch(10)` and
# 20 epochs. `LNNLoss`: 60 points `[q; q̇]` drawn from a normal distribution, with the accelerations
# that a first network draw predicts at them as the data, a second draw trained on them, `Batch(10)`
# and 30 epochs. Both use `LagrangianNeuralNetwork(2; width = 4, nhidden = 1, activation = softplus)`
# and Adam with its default step size. Each seed sets the global random number generator before the
# networks and the data are drawn.
#
# The script prints one line per loss, precision and seed, and exits 1 when the last loss is below
# the first on fewer than 14 of the 16 seeds for a loss and a precision. Under LNN_SMOKE it trains
# one seed for 2 epochs per loss and precision, and asserts nothing.
using LagrangianNeuralNetworks
using GeometricMachineLearning: CPU, Adam, Batch, DataLoader, NeuralNetwork,
                                TrainingOptimizer,
                                params, train!
import Random

# `smoke_size(full, smoke)` returns the second when LNN_SMOKE is set. See utilities/smoke.jl.
include("../utilities/smoke.jl")

const SEEDS = smoke_size(1:16, 1:1)
const REQUIRED = smoke_size(14, 0)
const VML_EPOCHS = smoke_size(20, 2)
const LNN_EPOCHS = smoke_size(30, 2)
const N_SAMPLES = 60
const Δt = 0.05

# One architecture and one loss of each kind for every run: each new one is a new compile.
const arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1, activation = softplus)
const vml = VariationalMidpointLoss(arch, Δt)
const lnn = LNNLoss(arch)

"Whether the last loss of `history` is below the first."
reduces(history) = history[end] < history[1]

"`VariationalMidpointLoss` on the triples of `cos t`, in precision `T`, under `seed`."
function vml_reduces(T::DataType, seed::Integer)
    Random.seed!(seed)
    q = cos.(T(Δt) .* (0:(N_SAMPLES + 1)))
    input = vcat(q[1:N_SAMPLES]', q[2:(N_SAMPLES + 1)]')
    output = collect(q[3:(N_SAMPLES + 2)]')
    nn = NeuralNetwork(arch, CPU(), T)
    dl = DataLoader(input, output; suppress_info = true)
    optimizer = TrainingOptimizer(nn.params; algorithm = Adam())
    reduces(train!(
        nn, optimizer, dl, Batch(10), VML_EPOCHS; loss = vml, show_progress = false))
end

"`LNNLoss` on the accelerations of a first network draw, in precision `T`, under `seed`."
function lnn_reduces(T::DataType, seed::Integer)
    Random.seed!(seed)
    first_draw = NeuralNetwork(arch, CPU(), T)
    input = randn(T, 2, N_SAMPLES)
    output = lnn.acceleration(input, params(first_draw))
    # The training starts from a second draw, so that the target is not already met.
    nn = NeuralNetwork(arch, CPU(), T)
    dl = DataLoader(input, output; suppress_info = true)
    optimizer = TrainingOptimizer(nn.params; algorithm = Adam())
    reduces(train!(
        nn, optimizer, dl, Batch(10), LNN_EPOCHS; loss = lnn, show_progress = false))
end

const RUNS = (("VariationalMidpointLoss", vml_reduces), ("LNNLoss", lnn_reduces))

failed = String[]
for (name, reduces_on) in RUNS, T in (Float32, Float64)

    count = 0
    for seed in SEEDS
        reduced = reduces_on(T, seed)
        count += reduced
        println(name, "  ", T, "  seed ", seed, "  last loss below the first: ", reduced)
        flush(stdout)
    end
    println(name, "  ", T, ": ", count, " of ", length(SEEDS), " seeds")
    count < REQUIRED && push!(failed, string(name, " in ", T))
end

if isempty(failed)
    println("the last loss is below the first on at least ", REQUIRED, " seeds of each loss and precision")
else
    println("fewer than ", REQUIRED, " seeds reduce the loss: ", join(failed, ", "))
    exit(1)
end
