@doc raw"""
    DiscreteLagrangian(nn, Δt)
    DiscreteLagrangian(nn)
    DiscreteLagrangian(D₁L_d, D₂L_d, D₁D₂L_d)

The discrete Lagrangian ``L_d(a, b)`` of two consecutive positions, the model that
[`variational_rollout`](@ref) integrates, in its three forms.

`DiscreteLagrangian(nn, Δt)` is the **midpoint** discrete Lagrangian of a `NeuralNetwork` of a
[`LagrangianArchitecture`](@ref),

```math
    L_d(a, b) = \Delta{}t\,L\!\left(\frac{a + b}{2}, \frac{b - a}{\Delta{}t}\right),
```

read through the chain rule as the two slot derivatives and the mixed Hessian

```math
    D_1L_d = \frac{\Delta{}t}{2}\nabla_qL - \nabla_{\dot{q}}L, \qquad
    D_2L_d = \frac{\Delta{}t}{2}\nabla_qL + \nabla_{\dot{q}}L, \qquad
    D_1D_2L_d = \frac{\Delta{}t}{4}H_{qq} + \frac{1}{2}\left(H_{q\dot{q}} - H_{\dot{q}q}\right)
        - \frac{1}{\Delta{}t}H_{\dot{q}\dot{q}}
        = -\frac{\Lambda}{\Delta{}t},
```

all at the midpoint, with ``\Lambda`` the ``-\Delta{}t\,D_1D_2L_d`` that
[`MixedHessianRegulariser`](@ref) builds from the same input Hessian. The timestep is converted to
the element type of the network's parameters.

`DiscreteLagrangian(nn)` is a `NeuralNetwork` of a [`DiscreteLagrangianArchitecture`](@ref), which
is a discrete Lagrangian already: its input `[a; b]` gives ``D_1L_d = g[1:n]``,
``D_2L_d = g[n+1:2n]`` and ``D_1D_2L_d = H[1:n, n+1:2n]`` as blocks of its own input gradient `g`
and input Hessian `H`.

`DiscreteLagrangian(D₁L_d, D₂L_d, D₁D₂L_d)` takes three functions of `(a, b)` — the two slot
derivatives as vectors of ``n`` entries and the mixed Hessian as an ``n\times{}n`` matrix, indexed
``[i, j] = \partial^2L_d/\partial{}a_i\partial{}b_j``. It names no precision and no dimension of its
own: a rollout takes both from the positions.

The slot derivatives and the momentum map ``p = -D_1L_d`` are those of Offen 2023, Prop. 1, and
Ober-Blöbaum & Offen 2023, §2.3. A timestep that is not positive and finite throws an
`ArgumentError`, and so does a bare continuous network, whose message names
`DiscreteLagrangian(nn, Δt)`.
"""
struct DiscreteLagrangian{T, N, D1T, D2T, D12T}
    D₁::D1T
    D₂::D2T
    D₁D₂::D12T
end

function DiscreteLagrangian(D₁, D₂, D₁D₂)
    DiscreteLagrangian{Nothing, Nothing, typeof(D₁), typeof(D₂), typeof(D₁D₂)}(D₁, D₂, D₁D₂)
end

# A model of a network names the element type of the network's parameters and the number of
# positions the network takes, which the three functions alone cannot tell.
function _discrete_lagrangian(T, n::Integer, d₁, d₂, d₁d₂)
    DiscreteLagrangian{T, n, typeof(d₁), typeof(d₂), typeof(d₁d₂)}(d₁, d₂, d₁d₂)
end

function DiscreteLagrangian(
        nn::AbstractNeuralNetworks.NeuralNetwork{<:LagrangianArchitecture}, Δt::Real)
    Δt = _check_timestep(Δt)
    arch = nn.architecture
    ps = nn.params
    n = dim(arch) ÷ 2
    δ = parameter_eltype(ps)(Δt)
    D = LagrangianDerivatives(arch)
    _discrete_lagrangian(parameter_eltype(ps), n,
        (a, b) -> begin
            ∇qL, ∇q̇L = _midpoint_gradient(D, a, b, δ, n, ps)
            (δ / 2) * ∇qL - ∇q̇L
        end,
        (a, b) -> begin
            ∇qL, ∇q̇L = _midpoint_gradient(D, a, b, δ, n, ps)
            (δ / 2) * ∇qL + ∇q̇L
        end,
        (a, b) -> -_mixed_hessian(_midpoint_hessian(D, a, b, δ, n, ps), δ) / δ)
end

function DiscreteLagrangian(
        nn::AbstractNeuralNetworks.NeuralNetwork{<:DiscreteLagrangianArchitecture})
    arch = nn.architecture
    ps = nn.params
    n = dim(arch) ÷ 2
    D = LagrangianDerivatives(arch)
    _discrete_lagrangian(parameter_eltype(ps), n,
        (a, b) -> input_gradient(D, _pair_input(a, b, n), ps)[1:n, 1],
        (a, b) -> input_gradient(D, _pair_input(a, b, n), ps)[(n + 1):(2n), 1],
        (a, b) -> input_hessian(D, _pair_input(a, b, n), ps)[1:n, (n + 1):(2n), 1])
end

function DiscreteLagrangian(nn::AbstractNeuralNetworks.NeuralNetwork{<:LagrangianArchitecture})
    throw(ArgumentError("a continuous Lagrangian needs a timestep: build `DiscreteLagrangian(nn, Δt)`"))
end

# The pair `[a; b]` as a one-column batch, the input shape of a discrete Lagrangian network.
_pair_input(a, b, n) = reshape(vcat(a, b), 2n, 1)

# The gradient of the continuous Lagrangian at the midpoint of `(a, b)`, split into its position
# and its velocity blocks.
function _midpoint_gradient(D::LagrangianDerivatives, a, b, δ, n, ps)
    midpoint = _pair_input((a + b) / 2, (b - a) / δ, n)
    g = input_gradient(D, midpoint, ps)
    g[1:n, 1], g[(n + 1):(2n), 1]
end

# The input Hessian of the continuous Lagrangian at the midpoint of `(a, b)`.
function _midpoint_hessian(D::LagrangianDerivatives, a, b, δ, n, ps)
    midpoint = _pair_input((a + b) / 2, (b - a) / δ, n)
    input_hessian(D, midpoint, ps)[:, :, 1]
end

# The element type of a rollout's positions: the parameters' for a network, and the positions' for
# a model of three functions, which fixes no precision of its own.
_precision(::DiscreteLagrangian{T}, ::AbstractVector, ::AbstractVector) where {T} = T
function _precision(::DiscreteLagrangian{Nothing}, q₀::AbstractVector, q₁::AbstractVector)
    float(promote_type(eltype(q₀), eltype(q₁)))
end

# The number of positions a model takes, or `nothing` when it is a model of functions, whose
# dimension is the length of the positions it is rolled out from.
_dimension(::DiscreteLagrangian{T, Nothing}) where {T} = nothing
_dimension(::DiscreteLagrangian{T, N}) where {T, N} = N

function _check_positions(model::DiscreteLagrangian, q₀::AbstractVector, q₁::AbstractVector)
    length(q₀) == length(q₁) ||
        throw(DimensionMismatch("q₀ has $(length(q₀)) entries and q₁ has $(length(q₁))"))
    n = _dimension(model)
    (n === nothing || length(q₀) == n) ||
        throw(DimensionMismatch("the positions have $(length(q₀)) entries, and the model takes $n"))
    nothing
end

@doc raw"""
    variational_rollout(model, q₀, q₁, N; momentum = false, tol, maxiter = 20)
    variational_rollout(model, q₀, p₀, N; momentum = true, tol, maxiter = 20)

Roll a trajectory out of a discrete Lagrangian `model`: the `n × (N + 1)` matrix
`[q₀ q₁ … q_N]` of the positions `q₀` and `q₁` and the `N − 1` that follow them.

Each step solves the discrete Euler–Lagrange equations (Offen 2023, Prop. 1)

```math
    D_2L_d(q_{k-1}, q_k) + D_1L_d(q_k, q_{k+1}) = 0
```

for ``q_{k+1}`` by Newton's method, with ``D_1D_2L_d(q_k, \cdot)`` as the Jacobian and
``2q_k - q_{k-1}`` as the guess, until the 2-norm of the residual is at most `tol`. The positions
are numbered from zero, so the message of a failed solve names the position it was solving for as
`q_k` with that index.

With `momentum = true` the second argument is the momentum ``p_0 = -D_1L_d(q_0, q_1)`` rather than
a position: `q₁` is recovered first, by solving ``-D_1L_d(q_0, q_1) = p_0`` from the guess `q₀`,
and the rollout then proceeds from the pair. Both forms take four positional arguments of the same
types, which is why the keyword selects this one. `N = 0` returns `q₀` alone, as one column, and
solves nothing.

`tol` defaults to ``\sqrt{\epsilon(T)}``, absolute and in the units of the residual, which scale
with the Lagrangian. `maxiter` counts Newton steps, and the residual is tested before the first
step and after each one; a residual that is not finite stops the solve at once, and one that does
not reach `tol` in `maxiter` steps throws an `ErrorException` naming the position.

The element type `T` of the result is the parameters' for a network model and
`float(promote_type(eltype(q₀), eltype(q₁)))` for a model of functions, and the positions are
converted to it. `model` is a [`DiscreteLagrangian`](@ref) or a `NeuralNetwork` of a
[`DiscreteLagrangianArchitecture`](@ref); a continuous network throws an `ArgumentError` naming
`DiscreteLagrangian(nn, Δt)`.

`N < 0`, a `tol` that is not positive and a `maxiter` below one throw an `ArgumentError`; positions
of another length than each other or than the network's throw a `DimensionMismatch`; a singular
Jacobian throws a `LinearAlgebra.SingularException`.
"""
function variational_rollout(model::DiscreteLagrangian, q₀::AbstractVector,
        q₁::AbstractVector, N::Integer;
        momentum::Bool = false, tol = nothing, maxiter::Integer = 20)
    N < 0 && throw(ArgumentError("N must be non-negative, got $N"))
    maxiter < 1 && throw(ArgumentError("maxiter must be at least one, got $maxiter"))
    _check_positions(model, q₀, q₁)
    T = _precision(model, q₀, q₁)
    tolerance = tol === nothing ? sqrt(eps(T)) : tol
    tolerance > zero(tolerance) ||
        throw(ArgumentError("tol must be positive, got $tolerance"))

    n = length(q₀)
    Q = Matrix{T}(undef, n, N + 1)
    Q[:, 1] .= q₀
    N == 0 && return Q

    # The positions a solve reads are the result's columns, so that the element type of the network
    # is the one the solve runs in whatever the caller passed.
    position = T.(q₁)
    if momentum
        Q[:, 2] .= _solve_momentum(model, Q[:, 1], position, tolerance, maxiter)
    else
        Q[:, 2] .= position
    end
    for k in 1:(N - 1)
        c = _solve_next(model, Q[:, k], Q[:, k + 1], tolerance, maxiter, k + 1)
        Q[:, k + 2] .= c
    end
    Q
end

function variational_rollout(model, q₀::AbstractVector, q₁::AbstractVector, N::Integer;
        momentum::Bool = false, tol = nothing, maxiter::Integer = 20)
    variational_rollout(
        _discrete_model(model), q₀, q₁, N; momentum, tol, maxiter)
end

# The model a rollout integrates: a `DiscreteLagrangian` as it stands, a discrete network as the
# discrete Lagrangian it already is, and a continuous network refused, since the midpoint rule that
# would turn it into one needs a timestep.
_discrete_model(model::DiscreteLagrangian) = model
function _discrete_model(nn::AbstractNeuralNetworks.NeuralNetwork{<:DiscreteLagrangianArchitecture})
    DiscreteLagrangian(nn)
end
function _discrete_model(nn::AbstractNeuralNetworks.NeuralNetwork{<:LagrangianArchitecture})
    throw(ArgumentError("a continuous Lagrangian needs a timestep: build `DiscreteLagrangian(nn, Δt)`"))
end

# Newton's method on one residual, with a residual test before the first step and after each one.
# A residual that is not finite stops at once, since the step it would ask for is not defined.
function _newton(residual::F, jacobian::J, guess, tol, maxiter::Integer, label) where {F, J}
    q = guess
    for step in 0:maxiter
        r = residual(q)
        all(isfinite, r) ||
            throw(ErrorException("the discrete Euler–Lagrange residual at $label is not finite"))
        norm(r) ≤ tol && return q
        step == maxiter && break
        q = q - jacobian(q) \ r
    end
    throw(ErrorException(
        "the discrete Euler–Lagrange solve for $label did not reach the tolerance in $maxiter Newton steps"))
end

# The position `c` that solves `D₂L_d(a, b) + D₁L_d(b, c) = 0`, from the guess `2b − a`.
function _solve_next(model::DiscreteLagrangian, a, b, tol, maxiter::Integer, index::Integer)
    residual(c) = model.D₂(a, b) + model.D₁(b, c)
    jacobian(c) = model.D₁D₂(b, c)
    _newton(residual, jacobian, 2b - a, tol, maxiter, _position_name(index))
end

# The position `b` that solves the momentum map `−D₁L_d(a, b) = p`, from the guess `a`.
function _solve_momentum(model::DiscreteLagrangian, a, p, tol, maxiter::Integer)
    residual(b) = model.D₁(a, b) + p
    jacobian(b) = model.D₁D₂(a, b)
    _newton(residual, jacobian, copy(a), tol, maxiter, _position_name(1))
end

# The name of a position in a message: the positions are numbered from zero, and a message names
# one as `q_` followed by its index.
_position_name(index::Integer) = "q_$index"
