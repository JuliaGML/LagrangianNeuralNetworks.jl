# Comparison

The four methods of the package, each trained with one fixed protocol on five benchmarks and
measured with the package's measures of a learned model, which do not depend on its gauge. The
tables below are the committed results of the script `scripts/comparison/compare.jl`: the page
reads them when the manual is built, and trains nothing.

```@setup comparison
using LagrangianNeuralNetworks
using Markdown: Markdown
using Printf: @sprintf
using TOML: TOML

const RESULTS = joinpath(pkgdir(LagrangianNeuralNetworks), "scripts", "comparison", "results")
const PROVENANCE = TOML.parsefile(joinpath(RESULTS, "provenance.toml"))

# The name of each method of the CSV files, as the page shows it.
const METHOD_NAMES = Dict("lnn" => "LNNLoss", "euler_lagrange" => "EulerLagrangeLoss",
    "variational_midpoint" => "VariationalMidpointLoss",
    "discrete_lagrangian" => "DiscreteLagrangianNetwork")

const COLUMNS = ["method", "acceleration error", "energy alignment", "trajectory error",
    "training time (s)", "final loss", "status"]

# A number of the CSV to three significant digits, and `—` where the run has no value.
number(s) = (x = parse(Float64, s); isnan(x) ? "—" : @sprintf("%.3g", x))

# The table of one benchmark: one row per line of its CSV, in the order of the file.
row_line(cells) = "| " * join(cells, " | ") * " |"
function comparison_table(name)
    rows = map(readlines(joinpath(RESULTS, "$name.csv"))[2:end]) do line
        c = split(line, ",")
        row_line(["`$(METHOD_NAMES[c[1]])`"; number.(c[2:6]); c[7]])
    end
    rule = "|:--|" * repeat("--:|", 5) * ":--|"
    Markdown.parse(join([row_line(COLUMNS); rule; rows], "\n"))
end

# The commit, the date, the Julia version, the command and the training time of the run.
run_statement() = Markdown.parse("""
The results come from commit `$(PROVENANCE["commit"][1:7])` of the repository, made on
$(PROVENANCE["date"]) with Julia $(PROVENANCE["julia"]) by the command
`$(PROVENANCE["command"])`, run from the root of the repository; the training times of the rows
sum to $(round(Int, PROVENANCE["total_training_time"])) seconds.
""")
```

## The protocol

Every benchmark is a Lagrangian ``L(q, \dot{q})`` with ``L(0) = 0``, ``\partial_{\dot{q}}L(0) = 0``
and ``\det\partial^2L/\partial\dot{q}^2(0) = 1``:

| benchmark | ``L`` | ``n`` | box of ``[q; \dot{q}]`` | ``\Delta{}t`` | rollout from |
|:--|:--|--:|:--|--:|:--|
| harmonic oscillator | ``(\dot{q}^2 - q^2)/2`` | 1 | ``[-1, 1]^2`` | 0.1 | ``[0.5, 0]`` |
| coupled oscillator | ``\frac12\lVert\dot{q}\rVert^2 - \frac12\lVert{}q\rVert^2 + 0.1\,q_1q_2`` | 2 | ``[-1, 1]^4`` | 0.1 | ``[0.5, -0.3, 0, 0]`` |
| pendulum | ``\dot{q}^2/2 - (1 - \cos q)`` | 1 | ``[-1.2, 1.2]^2`` | 0.1 | ``[0.8, 0]`` |
| Hénon–Heiles | ``\frac12\lVert\dot{q}\rVert^2 - \frac12\lVert{}q\rVert^2 - 0.8(q_1^2q_2 - q_2^3/3)`` | 2 | ``[-0.4, 0.4]^4`` | 0.1 | ``[0.3, -0.2, 0, 0.1]`` |
| double pendulum | ``\dot{q}_1^2 + \frac12\dot{q}_2^2 + \dot{q}_1\dot{q}_2\cos(q_1 - q_2) - 2(1 - \cos q_1) - (1 - \cos q_2)`` | 2 | ``[-1, 1]^4`` | 0.05 | ``[0.5, -0.5, 0, 0]`` |

Each network has one hidden layer of width 4 and the activation `softplus`, in `Float64`, and is
trained with `Adam` at the step size `linesearch = 1e-2`, `Batch(20)`, for 200 epochs, from seed
1234; each method sets the seed again before it builds its network, so a row does not depend on
the rows before it. The acceleration data are the ``21 \times 21`` grid of the box for ``n = 1``
and 400 uniform samples of it for ``n = 2``, with the closed-form acceleration of the benchmark.
The position data are 10 trajectories of 50 steps for ``n = 1`` and 20 of 20 steps for ``n = 2``,
from uniform states of the box, computed with the classical Runge–Kutta method at the step
``\Delta{}t/20`` and kept every ``\Delta{}t``.

The four methods:

- `LNNLoss`: `LagrangianNeuralNetwork(2n; width)` on the acceleration data;
- `EulerLagrangeLoss`: the same network wrapped in `NormalisedLagrangian` with `base = zeros(2n)`,
  `value = 0` and `momentum = zeros(n)`, with the loss `EulerLagrangeLoss` plus `ScaleCondition`
  at `zeros(2n)` with the value 1, of weight 1, on the acceleration data;
- `VariationalMidpointLoss`: `LagrangianNeuralNetwork(2n; width)` with its default regulariser
  `MixedHessianRegulariser(arch, Δt; form = :tamed)` of weight 1, on the triples
  ``([q_k; q_{k+1}], q_{k+2})`` of the position data;
- `DiscreteLagrangianNetwork`: `DiscreteLagrangianNetwork(n; width)` with
  `DiscreteEulerLagrangeLoss(arch, Δt)` and its default regulariser, on the same triples.

The columns of a table:

- **acceleration error**: `acceleration_error`, the relative error of the acceleration on the
  ``21 \times 21`` grid of the box (``n = 1``) or its ``5^4`` grid (``n = 2``);
- **energy alignment**: `energy_alignment`, the mean ``|\sin\theta|`` of the angle between the
  gradients of the learned and of the true energy on the same grid, 0 where their level sets agree;
- **trajectory error**: `trajectory_error`, the largest distance between a `variational_rollout`
  of 100 steps of ``\Delta{}t`` from the first two positions of the reference trajectory and that
  trajectory;
- **training time (s)**: the seconds from the construction of the loss to the end of `train!`;
- **final loss**: the loss of the last epoch, whose scale differs from one loss to another;
- **status**: `ok`; `diverged` where the rollout fails, with no trajectory error; `failed` where
  the training or the measuring fails, with no acceleration error, energy alignment, trajectory
  error or final loss.

A continuous network is rolled out as the midpoint discrete Lagrangian
`DiscreteLagrangian(nn, Δt)`. The `DiscreteLagrangianNetwork` is rolled out as itself, and measured
through `inverse_modified_lagrangian(nn, Δt)`, whose gradient and Hessian the script takes by
central differences of step ``10^{-4}``. A `—` marks a value the run does not have.

## The run

```@example comparison
run_statement() # hide
```

## Harmonic oscillator

```@example comparison
comparison_table("harmonic_oscillator") # hide
```

## Coupled oscillator

```@example comparison
comparison_table("coupled_oscillator") # hide
```

## Pendulum

```@example comparison
comparison_table("pendulum") # hide
```

## Hénon–Heiles

```@example comparison
comparison_table("henon_heiles") # hide
```

## Double pendulum

```@example comparison
comparison_table("double_pendulum") # hide
```
