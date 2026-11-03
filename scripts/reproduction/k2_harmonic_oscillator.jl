# Train a Lagrangian neural network with its default activation on the 1D harmonic oscillator,
# `L = (q̇² − q²)/2`, on 16 seeds in each precision.
#
# The data are 400 points `(q, q̇)` uniform in `[-1, 1]²` with `q̈ = −q`. Each seed sets the global
# random number generator before the data and the network are drawn. The network is
# `LagrangianNeuralNetwork(2; width = 4)`, trained with `LNNLoss`, Adam with a step size of `1e-2`,
# `Batch(20)` and 200 epochs. A loss of 1 is the loss of the prediction `q̈ = 0`.
#
# The script prints one line per seed and precision, and exits 1 when fewer than 14 of the 16 last
# losses of a precision are below 0.25. One training takes minutes, so the full run takes over an
# hour. Under LNN_SMOKE it trains one seed for 2 epochs per precision and asserts nothing.
using LagrangianNeuralNetworks
using GeometricMachineLearning: CPU, Adam, Batch, DataLoader, NeuralNetwork,
                                TrainingOptimizer,
                                train!
import Random

# `smoke_size(full, smoke)` returns the second when LNN_SMOKE is set. See utilities/smoke.jl.
include("../utilities/smoke.jl")

const SEEDS = smoke_size(1:16, 1:1)
const N_EPOCHS = smoke_size(200, 2)
const N_SAMPLES = 400
const BOUND = 0.25
const REQUIRED = smoke_size(14, 0)

# One architecture and one loss for every run: each new one is a new compile.
const arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1)
const loss = LNNLoss(arch)

"The last entry of the loss history of one training in precision `T` under `seed`."
function last_loss(T::DataType, seed::Integer)
    Random.seed!(seed)
    input = 2 .* rand(T, 2, N_SAMPLES) .- 1
    output = -input[1:1, :]
    nn = NeuralNetwork(arch, CPU(), T)
    dl = DataLoader(input, output; suppress_info = true)
    optimizer = TrainingOptimizer(nn.params; algorithm = Adam(), linesearch = T(1.0e-2))
    history = train!(nn, optimizer, dl, Batch(20), N_EPOCHS; loss, show_progress = false)
    history[end]
end

failed = String[]
for T in (Float32, Float64)
    below = 0
    for seed in SEEDS
        last = last_loss(T, seed)
        below += last < BOUND
        println(T, "  seed ", seed, "  last loss ", last)
        flush(stdout)
    end
    println(T, ": ", below, " of ", length(SEEDS), " last losses below ", BOUND)
    below < REQUIRED && push!(failed, string(T))
end

if isempty(failed)
    println("K2: at least ", REQUIRED, " last losses of each precision below ", BOUND)
else
    println("K2: fewer than ", REQUIRED, " seeds below ", BOUND, " in ", join(failed, ", "))
    exit(1)
end
