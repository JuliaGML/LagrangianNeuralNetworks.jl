# Train a Lagrangian neural network with `VariationalMidpointLoss` on the seeds 1 to 16 in each
# precision, and count the seeds on which the loss on the whole data set, at the parameters after
# the training, is below the same loss at the parameters before it.
#
# The loss has its default regulariser and weight. The data are the triples
# `(q(t), q(t + Δt), q(t + 2Δt))` of `q(t) = cos t` at `Δt = 0.05`, 60 of them, trained with
# `Batch(10)` for 20 epochs. The network is
# `LagrangianNeuralNetwork(2; width = 4, nhidden = 1, activation = softplus)`, and the optimiser is
# Adam with its default step size. Each seed sets the global random number generator before the
# network is drawn. The script checks no training with `LNNLoss`; K10 in KNOWN_ISSUES.md says why.
#
# The script prints one line per precision and seed, and exits 1 when the loss after the training
# is below the loss before it on fewer than 14 of the 16 seeds in a precision. Under LNN_SMOKE it
# trains one seed for 2 epochs per precision, and asserts nothing.
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
const N_SAMPLES = 60
const Δt = 0.05

# One architecture and one loss for every run: each new one is a new compile.
const arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1, activation = softplus)
const vml = VariationalMidpointLoss(arch, Δt)

"""
Train `nn` with `loss` for `epochs` epochs, and return whether the loss on the whole data set
`(input, output)` is below its value at the parameters before the training.
"""
function reduces(nn, loss, input, output, epochs)
    initial = loss(params(nn), input, output)
    dl = DataLoader(input, output; suppress_info = true)
    optimizer = TrainingOptimizer(nn.params; algorithm = Adam())
    train!(nn, optimizer, dl, Batch(10), epochs; loss = loss, show_progress = false)
    final = loss(params(nn), input, output)
    final < initial
end

"`VariationalMidpointLoss` on the triples of `cos t`, in precision `T`, under `seed`."
function vml_reduces(T::DataType, seed::Integer)
    Random.seed!(seed)
    q = cos.(T(Δt) .* (0:(N_SAMPLES + 1)))
    input = vcat(q[1:N_SAMPLES]', q[2:(N_SAMPLES + 1)]')
    output = collect(q[3:(N_SAMPLES + 2)]')
    nn = NeuralNetwork(arch, CPU(), T)
    reduces(nn, vml, input, output, VML_EPOCHS)
end

failed = String[]
for T in (Float32, Float64)
    count = 0
    for seed in SEEDS
        reduced = vml_reduces(T, seed)
        count += reduced
        println("VariationalMidpointLoss  ", T, "  seed ", seed,
            "  loss after the training below the loss before: ", reduced)
        flush(stdout)
    end
    println("VariationalMidpointLoss  ", T, ": ", count, " of ", length(SEEDS), " seeds")
    count < REQUIRED && push!(failed, string("VariationalMidpointLoss in ", T))
end

if isempty(failed)
    println(
        "the loss after the training is below the loss before it on at least ", REQUIRED,
        " seeds in each precision")
else
    println("fewer than ", REQUIRED, " seeds reduce the loss: ", join(failed, ", "))
    exit(1)
end
