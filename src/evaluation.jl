# The gauge-invariant measures of a learned Lagrangian (Offen 2025, §7.3): the energy and its
# gradient, the relative acceleration error, the alignment of two energies, and the error of a
# rolled-out trajectory. A Lagrangian is unique at best up to a scale, a constant and a total time
# derivative, so a comparison of two models goes through what they do and not through their values.

@doc raw"""
    LagrangianEnergy

The energy of a continuous Lagrangian and its gradient, as [`energy`](@ref) returns them. The
object is called as `E(q, q̇)` for the value, and its field `gradient` as `gradient(x)` for the
gradient of the energy at the state `x = [q; q̇]`.
"""
struct LagrangianEnergy{VT, GT}
    value::VT
    gradient::GT
end

(E::LagrangianEnergy)(q, q̇) = E.value(q, q̇)

@doc raw"""
    energy(model)

The energy of the continuous Lagrangian of `model`, a callable `E(q, q̇) = q̇ᵀ∂_q̇L − L` of two
vectors of ``n`` entries, with the element type of the network's parameters.

It is the quantity a trajectory conserves when the Lagrangian does not depend on time, and it is
the measure of a learned model that the level sets of *Evaluation* compare (Offen 2025, §7.3). A
Lagrangian is unique only up to a scale, a constant and a total time derivative, and the energy
changes with exactly the first two: under ``L \mapsto \rho L + c + \frac{d}{dt}F(q)`` it becomes
``\rho E - c``. Its value therefore carries the gauge, and its *gradient* does not.

The field `gradient` of the result is the gradient of the energy with respect to the state
``x = [q; \dot{q}]``,

```math
    \nabla E = H\begin{pmatrix}0\\ \dot{q}\end{pmatrix}
        - \begin{pmatrix}\nabla_qL\\ 0\end{pmatrix},
```

from the compiled input gradient ``\nabla_qL`` and the input Hessian ``H`` of the network, which is
what [`energy_alignment`](@ref) compares. It takes a vector of ``2n`` entries.

`energy` needs a continuous Lagrangian, and a [`DiscreteLagrangianArchitecture`](@ref) throws an
`ArgumentError`. A `q` or a `q̇` of another length throws a `DimensionMismatch`.
"""
function energy(model::AbstractNeuralNetworks.NeuralNetwork{<:LagrangianArchitecture})
    arch = model.architecture
    ps = model.params
    n = dim(arch) ÷ 2
    T = parameter_eltype(ps)
    D = LagrangianDerivatives(arch)
    LagrangianEnergy(
        (q::AbstractVector, q̇::AbstractVector) -> begin
            x = reshape(_state(q, q̇, n), 2n, 1)
            g = input_gradient(D, x, ps)
            T(dot(q̇, g[(n + 1):(2n), 1]) - only(model(x, ps)))
        end,
        (x::AbstractVector) -> begin
            size(x, 1) == 2n || throw(DimensionMismatch(
                "the state has $(size(x, 1)) entries, and the Lagrangian takes $(2n)"))
            xm = reshape(x, 2n, 1)
            g = input_gradient(D, xm, ps)
            H = input_hessian(D, xm, ps)
            state = vcat(zeros(T, n), x[(n + 1):(2n)])
            H[:, :, 1] * state - vcat(g[1:n, 1], zeros(T, n))
        end)
end

function energy(model::AbstractNeuralNetworks.NeuralNetwork{<:DiscreteLagrangianArchitecture})
    throw(ArgumentError(
        "energy needs a continuous Lagrangian: a discrete one is a function of two positions, and has no velocity to differentiate"))
end

# The state `[q; q̇]` of one point, once the length of both halves is the Lagrangian's.
function _state(q::AbstractVector, q̇::AbstractVector, n::Integer)
    length(q) == n ||
        throw(DimensionMismatch("q has $(length(q)) entries, and the Lagrangian takes $n"))
    length(q̇) == n ||
        throw(DimensionMismatch("q̇ has $(length(q̇)) entries, and the Lagrangian takes $n"))
    vcat(q, q̇)
end

@doc raw"""
    acceleration_error(model, x, q̈_ref)

The relative acceleration error ``\|\ddot{q}_{NN} - \ddot{q}_{ref}\| / \|\ddot{q}_{ref}\|`` of a
learned Lagrangian, over the columns of `x` and of `q̈_ref` (Offen 2025, §7.3).

`x` is the state ``[q; \dot{q}]`` of each sample, a ``2n\times{}B`` matrix, and `q̈_ref` the
reference acceleration, an ``n\times{}B`` matrix. ``\ddot{q}_{NN}`` is the acceleration the Euler–
Lagrange equations of the network predict,
``(\nabla_{\dot{q}}\nabla_{\dot{q}}L)^{-1}(\nabla_qL - (\nabla_q\nabla_{\dot{q}}L)^T\dot{q})``,
which is [`lagrangian_acceleration`](@ref); both norms are those of the whole ``n\times{}B`` array.

The relative error is invariant under the gauge of the data's model: it changes with the reference
and with nothing else about the Lagrangian, because a scale cancels between the residual and the
denominator. `model` is a `NeuralNetwork` of a [`LagrangianArchitecture`](@ref), and one of a
[`DiscreteLagrangianArchitecture`](@ref) throws an `ArgumentError`. `x` of the wrong row count, and
an acceleration of the wrong row or column count, throw a `DimensionMismatch`; no columns, and a
`q̈_ref` whose norm is zero, throw an `ArgumentError`; and a singular ``\nabla_{\dot{q}}
\nabla_{\dot{q}}L`` throws a `LinearAlgebra.SingularException`.
"""
function acceleration_error(
        model::AbstractNeuralNetworks.NeuralNetwork{<:LagrangianArchitecture},
        x::AbstractMatrix, q̈_ref::AbstractMatrix)
    arch = model.architecture
    n = dim(arch) ÷ 2
    size(x, 1) == 2n || throw(DimensionMismatch(
        "x has $(size(x, 1)) rows, and the Lagrangian takes $(2n)"))
    size(q̈_ref, 1) == n || throw(DimensionMismatch(
        "the reference acceleration has $(size(q̈_ref, 1)) rows, and the Lagrangian gives $n"))
    size(q̈_ref, 2) == size(x, 2) || throw(DimensionMismatch(
        "x has $(size(x, 2)) columns and the reference acceleration has $(size(q̈_ref, 2))"))
    size(x, 2) == 0 &&
        throw(ArgumentError("there are no points to measure the acceleration error at"))
    iszero(norm(q̈_ref)) && throw(ArgumentError(
        "the reference acceleration is zero, and a relative error against it is not defined"))
    norm(lagrangian_acceleration(arch)(x, model.params) - q̈_ref) / norm(q̈_ref)
end

function acceleration_error(
        model::AbstractNeuralNetworks.NeuralNetwork{<:DiscreteLagrangianArchitecture},
        x::AbstractMatrix, q̈_ref::AbstractMatrix)
    throw(ArgumentError(
        "acceleration_error needs a continuous Lagrangian: a discrete one is a function of two positions, and has no velocity to differentiate"))
end

@doc raw"""
    energy_alignment(E, E_ref, points)

The mean over the columns `x` of `points` of ``|\sin\theta|``, the angle between ``\nabla E(x)``
and ``\nabla E_{ref}(x)``: 1 where the two energies single out the same level sets at that point
and 0 where their gradients are parallel.

This is the measure ``\nu`` of Ober-Blöbaum & Offen 2023, §3.1, the mean of
``|\det(\nabla E_{ref}/\|\nabla E_{ref}\|, \nabla E/\|\nabla E\|)|``, which their two-dimensional
benchmarks use and which generalises to a state of ``2n > 2`` entries as the sine of the angle. It
is invariant under the gauge ``L \mapsto \rho L + c + \frac{d}{dt}F(q)``, which scales and shifts
``\nabla E`` by the same factors and sign at every point.

`points` is a ``2n\times{}B`` matrix of states. Each of `E` and `E_ref` is the result of
[`energy`](@ref), whose `gradient` is used, or a function `x ↦ ∇E(x)` of a state that returns the
gradient of length ``2n``: the energy of a benchmark has a closed-form gradient, and no dependency
of this package differentiates an arbitrary function.

A point whose gradient is exactly zero for either energy has no direction, and is left out of the
mean; when every point is, and when `points` has no column, an `ArgumentError` is thrown. The value
has the element type of the gradients.
"""
function energy_alignment(E, E_ref, points::AbstractMatrix)
    B = size(points, 2)
    B == 0 && throw(ArgumentError("there are no points to align the energies at"))
    U = _energy_gradients(E, points)
    V = _energy_gradients(E_ref, points)
    T = float(eltype(U))
    kept = 0
    total = zero(T)
    for b in 1:B
        u = U[:, b]
        v = V[:, b]
        # Each gradient is divided by its largest absolute entry before it is normalised: `norm`
        # overflows to `Inf` at the top of the range, where the direction is lost, and it rounds a
        # subnormal gradient to a scale of its own, where the quotient is not a unit vector. For a
        # finite gradient the scaled one has its largest entry at 1, so its norm lies between 1 and
        # `√(2n)`.
        mu = maximum(abs, u)
        iszero(mu) && continue
        u = u / mu
        mv = maximum(abs, v)
        iszero(mv) && continue
        v = v / mv
        nu, nv = norm(u), norm(v)
        # The sine as the part of the unit vector `v̂` orthogonal to the unit vector `û`, which stays
        # accurate where the two are nearly parallel, unlike a difference of two squared norms.
        û = u / nu
        v̂ = v / nv
        total += norm(v̂ - dot(û, v̂) * û)
        kept += 1
    end
    kept == 0 && throw(ArgumentError(
        "every point has a zero energy gradient, and its alignment is not defined"))
    total / kept
end

# The gradient of an energy at each column of `points`, as a `2n × B` matrix.
function _energy_gradients(E::LagrangianEnergy, points::AbstractMatrix)
    _gradient_columns(points, x -> E.gradient(x))
end
_energy_gradients(E, points::AbstractMatrix) = _gradient_columns(points, x -> E(x))

function _gradient_columns(points::AbstractMatrix, gradient)
    hcat((gradient(points[:, b]) for b in axes(points, 2))...)
end

@doc raw"""
    trajectory_error(Q, Q_ref)

The largest ``\|\cdot\|_2`` distance between the columns of two trajectories, `Q` and `Q_ref`,
each an ``n\times{}k`` matrix whose column ``k`` is the position at step ``k``.

It is the error of a rollout against a reference solution — the measure of a learned model that a
variational integrator needs, since a trajectory is what the model predicts and a position is what
it is compared at (Offen 2025, §7.3). The worst column is taken rather than a mean, because a
rollout that is right everywhere but one step is a rollout that has lost the trajectory.

Trajectories of different sizes throw a `DimensionMismatch` and empty ones an `ArgumentError`; a
`NaN` in either gives `NaN`. The value has the element type of the positions.
"""
function trajectory_error(Q::AbstractMatrix, Q_ref::AbstractMatrix)
    size(Q) == size(Q_ref) || throw(DimensionMismatch(
        "the trajectories differ in size: $(size(Q)) and $(size(Q_ref))"))
    size(Q, 2) == 0 && throw(ArgumentError("there are no columns to compare"))
    maximum(norm(Q[:, k] - Q_ref[:, k]) for k in axes(Q, 2))
end
