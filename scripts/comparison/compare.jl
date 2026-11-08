# Compare the four methods of the package over the five benchmarks of `benchmarks.jl`.
#
#     julia --startup-file=no --project=scripts scripts/comparison/compare.jl [outdir]
#
# For each benchmark and each method, in the orders of `BENCHMARKS` and `METHODS`, the script trains
# the method's network with one fixed protocol, measures it and writes one row of the benchmark's
# CSV; then it writes `provenance.toml`, with the command, the commit, whether `git status` names a
# file under `src/` or `scripts/`, the smoke switch, the Julia version, the date and the summed
# training time. `outdir` defaults to `results/` beside this script, the committed results that
# `docs/src/comparison.md` shows.
#
# The protocol, common to every benchmark: one hidden layer of `softplus`, `Float64`, `Adam` with
# `linesearch = 1e-2`, `Batch(20)`, 200 epochs and seed 1234. The data of a benchmark are drawn once
# after `Random.seed!(1234)`, and each method calls `Random.seed!(1234)` again before it builds its
# network, so a row does not depend on the rows before it. The accelerations are the closed forms of
# `benchmarks.jl`, and the positions come from its `rk4` with the step `Δt/20`, kept every `Δt`.
#
# The measures of a row: `acceleration_error` and `energy_alignment` on a grid of the benchmark's
# box, against the closed-form acceleration and the gradient of the true energy; `trajectory_error`
# of a `variational_rollout` of `N` steps of `Δt` from the first two positions of the reference
# trajectory from `x0`, against that trajectory; `training_time`, the seconds from the construction
# of the loss to the end of `train!`; and `final_loss`, the last entry of the loss history. The
# `DiscreteLagrangianNetwork` is measured through `inverse_modified_lagrangian(nn, Δt)`, whose
# gradient and Hessian the script takes by central differences, because `acceleration_error` and
# `energy` take a continuous Lagrangian only.
#
# A row's status is `failed` when its training, or the measuring on the grid, throws or is not
# finite, with the acceleration error, the energy alignment, the trajectory error and the final loss
# `NaN`; `diverged` when its rollout throws or is not finite, with `trajectory_error` `NaN`; and `ok`
# otherwise. A failed row does not stop the script.
#
# Under LNN_SMOKE every size is the smoke size (see utilities/smoke.jl), and the output directory is
# required, so that a smoke run never overwrites the committed results.

include(joinpath(@__DIR__, "..", "utilities", "smoke.jl"))

const SMOKE = smoke_size(false, true)

length(ARGS) ≤ 1 ||
    throw(ArgumentError("compare.jl takes at most one argument, the output directory, and got $(length(ARGS))"))
SMOKE && isempty(ARGS) &&
    throw(ArgumentError("LNN_SMOKE is set: pass an output directory, so that the smoke run does not overwrite the committed results"))

using LagrangianNeuralNetworks
using GeometricMachineLearning: CPU, Adam, Batch, DataLoader, NeuralNetwork,
                                TrainingOptimizer, train!
using Random: Random
using TOML: TOML
using Zygote: Zygote

include(joinpath(@__DIR__, "benchmarks.jl"))

# The root of the tree, which the command and the git queries of the provenance are relative to.
const ROOT = rstrip(normpath(joinpath(@__DIR__, "..", "..")), '/')
const OUTDIR = isempty(ARGS) ? joinpath(@__DIR__, "results") : abspath(ARGS[1])

const METHODS = ["lnn", "euler_lagrange", "variational_midpoint", "discrete_lagrangian"]
const HEADER = "method,acceleration_error,energy_alignment,trajectory_error,training_time,final_loss,status"

const SEED = 1234
const EPOCHS = smoke_size(200, 2)
const BATCH_SIZE = 20
const STEP_SIZE = 1e-2
const SUBSTEPS = 20               # RK4 steps per `Δt` of the reference positions
const FD_STEP = 1e-4              # the central differences of the discrete network's measures
const ROLLOUT_STEPS = smoke_size(100, 5)
const WIDTH = b -> smoke_size(b.width, 2)

# The sizes that depend on the number of positions `n`, indexed by `n`.
const ACCELERATION_GRID = smoke_size(21, 5)         # points per axis of the grid, n = 1
const ACCELERATION_SAMPLES = smoke_size(400, 50)    # uniform samples of the box, n = 2
const TRAJECTORIES = (smoke_size(10, 2), smoke_size(20, 2))
const TRAJECTORY_STEPS = (smoke_size(50, 5), smoke_size(20, 5))
const MEASURE_GRID = (smoke_size(21, 5), smoke_size(5, 3))  # points per axis

"""
    box_grid(b, m)

The `2n × m^(2n)` matrix of the states of the grid of `m` points per axis of the box of `b`.
"""
function box_grid(b, m)
    axes_ = (range(b.lower[i], b.upper[i]; length = m) for i in 1:(2b.n))
    reduce(hcat, [collect(x) for x in vec(collect(Iterators.product(axes_...)))])
end

"The `2n × k` matrix of `k` states drawn uniformly from the box of `b`."
uniform_states(b, k) = b.lower .+ rand(2b.n, k) .* (b.upper .- b.lower)

"The closed-form accelerations of `b` at the columns of `X`, an `n × B` matrix."
function accelerations(b, X)
    reduce(hcat, [b.acceleration(x[1:(b.n)], x[(b.n + 1):end]) for x in eachcol(X)])
end

"The `n × (steps + 1)` reference positions of `b` from the state `x`, every `Δt`."
function reference_positions(b, x, steps)
    traj = rk4(
        b.acceleration, x[1:(b.n)], x[(b.n + 1):end], b.Δt / SUBSTEPS, SUBSTEPS * steps)
    traj.q[:, 1:SUBSTEPS:end]
end

"""
    benchmark_data(b)

The training data of `b`, drawn after `Random.seed!(SEED)`: `acceleration = ([q; q̇], q̈)` and
`positions = ([qₖ; qₖ₊₁], qₖ₊₂)`, the triples of trajectories that start at uniform states of the box.
"""
function benchmark_data(b)
    Random.seed!(SEED)
    X = b.n == 1 ? box_grid(b, ACCELERATION_GRID) : uniform_states(b, ACCELERATION_SAMPLES)
    starts = uniform_states(b, TRAJECTORIES[b.n])
    trajectories = [reference_positions(b, x, TRAJECTORY_STEPS[b.n])
                    for x in eachcol(starts)]
    input = reduce(hcat, [vcat(P[:, 1:(end - 2)], P[:, 2:(end - 1)]) for P in trajectories])
    output = reduce(hcat, [P[:, 3:end] for P in trajectories])
    (acceleration = (X, accelerations(b, X)), positions = (input, output))
end

"""
    energy_gradient(L, n)

The function `x ↦ ∇E(x)` of the energy `E = q̇ᵀ∂_q̇L − L` of the Lagrangian `L` of `x = [q; q̇]`:
`∇_qE = (∂²L/∂q∂q̇) q̇ − ∇_qL` and `∇_q̇E = (∂²L/∂q̇²) q̇`, with the mixed block indexed
`[i, j] = ∂²L/∂qᵢ∂q̇ⱼ`, from the gradient and the Hessian of `L` that `Zygote` takes.
"""
function energy_gradient(L, n)
    x -> begin
        g, H, q̇ = Zygote.gradient(L, x)[1], Zygote.hessian(L, x), x[(n + 1):(2n)]
        vcat(H[1:n, (n + 1):(2n)] * q̇ - g[1:n], H[(n + 1):(2n), (n + 1):(2n)] * q̇)
    end
end

"""
    central_derivatives(ℓ, x, h)

The gradient and the Hessian of the scalar function `ℓ` at `x`, by central differences of step `h`.
"""
function central_derivatives(ℓ, x, h)
    m = length(x)
    e(i) = (v = zeros(m); v[i] = h; v)
    g = [(ℓ(x + e(i)) - ℓ(x - e(i))) / 2h for i in 1:m]
    H = [(ℓ(x + e(i) + e(j)) - ℓ(x + e(i) - e(j)) - ℓ(x - e(i) + e(j)) +
          ℓ(x - e(i) - e(j))) /
         (4h^2) for i in 1:m, j in 1:m]
    g, H
end

"""
    discrete_measures(nn, b, points)

The acceleration error and the energy alignment of the `DiscreteLagrangianNetwork` `nn` on the
columns of `points`, through `inverse_modified_lagrangian(nn, Δt)`, whose gradient `g` and Hessian
`H` come from `central_derivatives`: the acceleration is `H_q̇q̇ \\ (g_q − H_qq̇ᵀ q̇)` and the energy
gradient is that of `energy_gradient`.
"""
function discrete_measures(nn, b, points)
    n = b.n
    L = inverse_modified_lagrangian(nn, b.Δt)
    ℓ = x -> L(x[1:n], x[(n + 1):(2n)])
    function derivatives(x)
        g, H = central_derivatives(ℓ, x, FD_STEP)
        g, H, x[(n + 1):(2n)]
    end
    function acceleration(x)
        g, H, q̇ = derivatives(x)
        H[(n + 1):(2n), (n + 1):(2n)] \ (g[1:n] - H[1:n, (n + 1):(2n)]' * q̇)
    end
    function gradient(x)
        g, H, q̇ = derivatives(x)
        vcat(H[1:n, (n + 1):(2n)] * q̇ - g[1:n], H[(n + 1):(2n), (n + 1):(2n)] * q̇)
    end
    q̈ = reduce(hcat, [acceleration(x) for x in eachcol(points)])
    q̈_ref = accelerations(b, points)
    error = sqrt(sum(abs2, q̈ - q̈_ref)) / sqrt(sum(abs2, q̈_ref))
    error, energy_alignment(gradient, energy_gradient(b.lagrangian, n), points)
end

"""
    method_setup(method, b)

The architecture of `method` on the benchmark `b`, a function that builds its loss, and the
field of `benchmark_data` that it trains on.
"""
function method_setup(method, b)
    n, width, Δt = b.n, WIDTH(b), b.Δt
    if method == "lnn"
        arch = LagrangianNeuralNetwork(2n; width, nhidden = 1, activation = softplus)
        (arch = arch, loss = () -> LNNLoss(arch), data = :acceleration)
    elseif method == "euler_lagrange"
        inner = LagrangianNeuralNetwork(2n; width, nhidden = 1, activation = softplus)
        arch = NormalisedLagrangian(inner; base = zeros(2n), value = 0, momentum = zeros(n))
        loss = () -> RegularisedLoss(EulerLagrangeLoss(arch),
            1 => ScaleCondition(arch; base = zeros(2n), value = 1))
        (arch = arch, loss = loss, data = :acceleration)
    elseif method == "variational_midpoint"
        arch = LagrangianNeuralNetwork(2n; width, nhidden = 1, activation = softplus)
        (arch = arch, loss = () -> VariationalMidpointLoss(arch, Δt), data = :positions)
    elseif method == "discrete_lagrangian"
        arch = DiscreteLagrangianNetwork(n; width, nhidden = 1, activation = softplus)
        (arch = arch, loss = () -> DiscreteEulerLagrangeLoss(arch, Δt), data = :positions)
    else
        throw(ArgumentError("unknown method $method"))
    end
end

"""
    measure(method, nn, b, points)

The acceleration error and the energy alignment of the trained `nn` on the columns of `points`.
"""
function measure(method, nn, b, points)
    method == "discrete_lagrangian" && return discrete_measures(nn, b, points)
    acceleration_error(nn, points, accelerations(b, points)),
    energy_alignment(energy(nn), energy_gradient(b.lagrangian, b.n), points)
end

"""
    rollout_error(method, nn, b, Q_ref)

The `trajectory_error` of a `variational_rollout` of `nn` from the first two columns of the
reference positions `Q_ref`, over as many steps as `Q_ref` has after its first column.
"""
function rollout_error(method, nn, b, Q_ref)
    model = method == "discrete_lagrangian" ? DiscreteLagrangian(nn) :
            DiscreteLagrangian(nn, b.Δt)
    Q = variational_rollout(model, Q_ref[:, 1], Q_ref[:, 2], size(Q_ref, 2) - 1)
    trajectory_error(Q, Q_ref)
end

"""
    run_row(method, b, data, points, Q_ref)

Train, measure and roll out `method` on `b`, and return its row as a named tuple of the CSV's
columns. A row that throws or is not finite gets its status, and no exception leaves the function.
"""
function run_row(method, b, data, points, Q_ref)
    training_time, stage = NaN, "training"
    try
        Random.seed!(SEED)
        setup = method_setup(method, b)
        nn = NeuralNetwork(setup.arch, CPU(), Float64)
        dl = DataLoader(getfield(data, setup.data)...; suppress_info = true)
        start = time()
        loss = setup.loss()
        optimizer = TrainingOptimizer(nn.params; algorithm = Adam(), linesearch = STEP_SIZE)
        history = train!(nn, optimizer, dl, Batch(BATCH_SIZE), EPOCHS;
            loss = loss, show_progress = false)
        training_time = time() - start
        final_loss = Float64(last(history))
        isfinite(final_loss) || error("the final loss is $final_loss")
        stage = "measuring"
        acc, align = measure(method, nn, b, points)
        isfinite(acc) && isfinite(align) ||
            error("the acceleration error is $acc and the energy alignment $align")
        trajectory = try
            rollout_error(method, nn, b, Q_ref)
        catch err
            @warn "$(b.name)/$method: the rollout throws" exception = (
                err, catch_backtrace())
            NaN
        end
        status = isfinite(trajectory) ? "ok" : "diverged"
        trajectory = isfinite(trajectory) ? trajectory : NaN
        (method = method, acceleration_error = acc, energy_alignment = align,
            trajectory_error = trajectory, training_time = training_time,
            final_loss = final_loss, status = status)
    catch err
        @warn "$(b.name)/$method failed while $stage" exception = (err, catch_backtrace())
        (method = method, acceleration_error = NaN, energy_alignment = NaN,
            trajectory_error = NaN, training_time = training_time, final_loss = NaN,
            status = "failed")
    end
end

csv_line(row) = join(string.(values(row)), ",")

"""
    git(args...)

The output of `git args...` in the root of the tree, without its trailing newline.
"""
git(args...) = readchomp(Cmd(`git $args`; dir = ROOT))

"""
    shown_path(path)

`path` relative to the root of the tree when it lies inside it, and absolute otherwise.
"""
function shown_path(path)
    path = abspath(path)
    startswith(path, ROOT * "/") ? relpath(path, ROOT) : path
end

"""
    command()

The command that runs this script as it runs now, from the root of the tree.
"""
function command()
    words = ["julia", "--startup-file=no",
        "--project=" * shown_path(dirname(Base.active_project())),
        shown_path(PROGRAM_FILE), ARGS...]
    (SMOKE ? "LNN_SMOKE=$(ENV["LNN_SMOKE"]) " : "") * join(words, " ")
end

# Read before any file is written, so that the results this run writes do not count as changes.
const COMMIT = git("rev-parse", "HEAD")
const DIRTY = !isempty(git("status", "--porcelain", "--", "src", "scripts"))

function main()
    mkpath(OUTDIR)
    total_time = 0.0
    for b in BENCHMARKS
        data = benchmark_data(b)
        points = box_grid(b, MEASURE_GRID[b.n])
        Q_ref = reference_positions(b, b.x0, ROLLOUT_STEPS)
        path = joinpath(OUTDIR, "$(b.name).csv")
        open(path, "w") do io
            println(io, HEADER)
            for method in METHODS
                row = run_row(method, b, data, points, Q_ref)
                isfinite(row.training_time) && (total_time += row.training_time)
                println(io, csv_line(row))
                flush(io)
                println(rpad(b.name, 22), rpad(method, 22), csv_line(row))
                flush(stdout)
            end
        end
    end
    provenance = Dict(
        "command" => command(),
        "commit" => COMMIT,
        "dirty" => DIRTY,
        "smoke" => SMOKE,
        "julia" => string(VERSION),
        "date" => Libc.strftime("%Y-%m-%dT%H:%M:%S%z", time()),
        "total_training_time" => total_time
    )
    open(io -> TOML.print(io, provenance; sorted = true), joinpath(OUTDIR, "provenance.toml"), "w")
    println("wrote the results to ", OUTDIR)
end

# Run when the file is the program, and only define the functions when another file includes it.
abspath(PROGRAM_FILE) == (@__FILE__) && main()
