using LagrangianNeuralNetworks
using LagrangianNeuralNetworks: DiscreteLagrangian, _solve_momentum, variational_rollout
using AbstractNeuralNetworks: NeuralNetwork
using LinearAlgebra: I, SingularException
using Test
import Random

include("../helpers/eltypes.jl")

Random.seed!(1234)

square(x) = x^2

"""
    oscillator(T)

The network of the harmonic oscillator's Lagrangian `L = (q̇² − q²)/2` in one dimension, exactly:
one layer of the two neurons `q` and `q̇` with the activation `x²`, and the output weights
`(−1/2, 1/2)`.
"""
function oscillator(T::DataType)
    arch = LagrangianNeuralNetwork(2; width = 2, nhidden = 0, activation = square)
    nn = NeuralNetwork(arch, T)
    nn.params.L1.W .= T[1 0; 0 1]
    nn.params.L1.b .= 0
    nn.params.L2.W .= T[-1 // 2 1 // 2]
    nn
end

"""
    coupled(T; γ = 3/10)

The network of `L = ½‖q̇‖² − ½‖q‖² + γ q¹q̇²` in two dimensions, exactly: the five neurons
`q¹`, `q²`, `q̇¹`, `q̇²` and `q¹ + q̇²` with the activation `x²`, and the output weights that give
the squares that combination, `−1/2`, `−1/2`, `1/2`, `1/2` and `γ/2` corrected for the two squares
that `q¹ + q̇²` contributes.
"""
function coupled(T::DataType; γ = T(3 // 10))
    arch = LagrangianNeuralNetwork(4; width = 5, nhidden = 0, activation = square)
    nn = NeuralNetwork(arch, T)
    nn.params.L1.W .= T[1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 1; 1 0 0 1]
    nn.params.L1.b .= 0
    nn.params.L2.W .= T[(-1 - γ)/2 -1 // 2 1 // 2 (1 - γ)/2 γ/2]
    nn
end

# The momentum and the two slot derivatives of `L = ½‖q̇‖² − ½‖q‖² + γ q¹q̇²` at the midpoint of
# the pair `(a, b)`, and the mixed Hessian of its midpoint discrete Lagrangian, from the chain rule.
function coupled_midpoint(a, b, δ, γ)
    m = (a .+ b) ./ 2
    v = (b .- a) ./ δ
    ∇qL = [-m[1] + γ * v[2], -m[2]]
    ∇q̇L = [v[1], v[2] + γ * m[1]]
    Hqq = -Matrix(I, 2, 2)
    Hqv = [zero(γ) γ; zero(γ) zero(γ)]
    Hvv = Matrix(I, 2, 2)
    ((δ / 2) * ∇qL - ∇q̇L, (δ / 2) * ∇qL + ∇q̇L,
        (δ / 4) * Hqq + (Hqv - transpose(Hqv)) / 2 - Hvv / δ)
end

# The midpoint discrete Lagrangian of the harmonic oscillator as three functions of the pair, at the
# timestep `δ`.
function oscillator_functions(δ)
    ((a, b) -> (a - b) / δ - δ * (a + b) / 4,
        (a, b) -> (b - a) / δ - δ * (a + b) / 4,
        (a, b) -> -(1 + δ^2 / 4) / δ * Matrix{eltype(a)}(I, length(a), length(a)))
end

# A `NeuralNetwork` of the same parameters in `Float64`, so that the finite differences are taken in
# the precision the network's own values are compared in.
function same_network(arch, nn, ::Type{T}) where {T}
    wide = NeuralNetwork(arch, T)
    for layer in keys(nn.params), name in keys(nn.params[layer])

        getproperty(getproperty(wide.params, layer), name) .= getproperty(getproperty(nn.params, layer), name)
    end
    wide
end

# A hand-set `DiscreteLagrangianNetwork` of `L_d(a, b) = (a + b)²`, which is exact: `D₁L_d` and `D₂L_d`
# are both `2(a + b)`, `D₁D₂L_d` is `2`, and the discrete Euler–Lagrange equations give
# `q_{k+2} = −q_k − 2q_{k+1}` in one Newton step from any guess.
function quadratic_discrete(T::DataType)
    nn = NeuralNetwork(
        DiscreteLagrangianNetwork(1; width = 3, nhidden = 0, activation = square), T)
    nn.params.L1.W .= T[1 0; 0 1; 1 1]
    nn.params.L1.b .= 0
    nn.params.L2.W .= T[0 0 1]
    nn
end

# ---------------------------------------------------------------- the discrete model

@testset "the midpoint of the coupled network ($T)" for T in REAL_ELTYPES
    nn = coupled(T)
    model = DiscreteLagrangian(nn, 1 // 10)
    δ = T(1 // 10)
    γ = T(3 // 10)

    for (a, b) in ((T[0.3, -0.2], T[0.1, 0.5]), (T[-1.0, 0.4], T[0.7, -0.9]),
        (T[0.0, 0.0], T[0.25, -0.35]))
        D₁, D₂, D₁D₂ = coupled_midpoint(a, b, δ, γ)
        tolerance = T === Float64 ? 1.0e-12 : 1.0e-5

        for (got, want) in ((model.D₁(a, b), D₁), (model.D₂(a, b), D₂))
            @test eltype(got) === T
            @test size(got) == (2,)
            @test maximum(abs.(got .- want)) ≤ tolerance
        end

        mixed = model.D₁D₂(a, b)
        @test eltype(mixed) === T
        @test size(mixed) == (2, 2)
        @test maximum(abs.(mixed .- D₁D₂)) ≤ tolerance
    end
end

# The mixed Hessian must be `∂_b D₁L_d` of the same pair, and a quadratic network cannot see a
# wrong midpoint in it: with the activation `x²` and no hidden layer the input Hessian is the same
# at every point, while `softplus` varies. The finite difference is taken in `Float64` from the
# same parameters, on `D₁L_d`, which the first testset checks against its closed form.
@testset "the mixed Hessian of a nonlinear network ($T)" for T in REAL_ELTYPES
    arch = LagrangianNeuralNetwork(4; width = 4, nhidden = 1, activation = softplus)
    nn = NeuralNetwork(arch, T)
    model = DiscreteLagrangian(nn, 1 // 10)
    plain = DiscreteLagrangian(same_network(arch, nn, Float64), 1 // 10)
    h = 1.0e-5
    tolerance = T === Float64 ? 1.0e-6 : 1.0e-3

    for (a, b) in ((T[0.3, -0.2], T[0.1, 0.5]), (T[-0.6, 0.4], T[0.25, -0.35]))
        D₁D₂ = model.D₁D₂(a, b)
        @test eltype(D₁D₂) === T
        af, bf = Float64.(a), Float64.(b)
        wanted = hcat([(plain.D₁(af, bf .+ h .* e) - plain.D₁(af, bf .- h .* e)) / (2h)
                       for e in eachcol(Matrix(I, 2, 2))]...)
        @test maximum(abs.(Float64.(D₁D₂) .- wanted)) ≤ tolerance
    end
end

@testset "the blocks of a discrete network ($T)" for T in REAL_ELTYPES
    arch = DiscreteLagrangianNetwork(2; width = 4)
    nn = NeuralNetwork(arch, T)
    model = DiscreteLagrangian(nn)
    @test model isa DiscreteLagrangian

    # The same parameters in `Float64`, so that the central differences are taken in the precision
    # that the network's own value is compared in.
    wide = same_network(arch, nn, Float64)
    @test wide.params.L3.W == Float64.(nn.params.L3.W)

    value(x) = only(wide(reshape(x, 4, 1)))
    basis(i) = Matrix(1.0I, 2, 2)[:, i]
    # The weights of a central difference at the offsets `±h` and `±2h`, whose truncation error is
    # `O(h⁴)`. The two slot derivatives take the second-order stencil of step `1e-5`, whose round-off
    # is `eps·|L|/h`; the mixed block takes this one of step `1e-3`, because a second difference of it
    # has a round-off of `eps·|L|/h²`, which at `1e-5` is `2e-6·|L|` and above the `1e-6` relative
    # tolerance this testset asserts.
    weights = ((-2, 1.0), (-1, -8.0), (1, 8.0), (2, -1.0))
    h = 1.0e-5
    h₂ = 1.0e-3
    # `∂²/∂a_i∂b_j` by the same stencil in each argument.
    function curl(x, y, i, j, step)
        total = 0.0
        for (k, wk) in weights, (l, wl) in weights

            shift_a = k * step * basis(i)
            shift_b = l * step * basis(j)
            total += wk * wl * value(vcat(x .+ shift_a, y .+ shift_b))
        end
        total / (144 * step^2)
    end
    tolerance = T === Float64 ? 1.0e-6 : 1.0e-3

    for (a, b) in ((T[0.3, -0.2], T[0.1, 0.5]), (T[-1.0, 0.4], T[0.7, -0.9]),
        (T[0.0, 0.0], T[0.25, -0.35]))
        D₁, D₂, D₁D₂ = model.D₁(a, b), model.D₂(a, b), model.D₁D₂(a, b)
        @test eltype(D₁) === T
        @test eltype(D₂) === T
        @test eltype(D₁D₂) === T

        af = Float64.(a)
        bf = Float64.(b)
        gradient_a = [(value(vcat(af .+ h .* basis(i), bf)) -
                       value(vcat(af .- h .* basis(i), bf))) / (2h)
                      for i in 1:2]
        gradient_b = [(value(vcat(af, bf .+ h .* basis(j))) -
                       value(vcat(af, bf .- h .* basis(j)))) / (2h)
                      for j in 1:2]
        hessian = [curl(af, bf, i, j, h₂) for i in 1:2, j in 1:2]

        for (got, want) in ((D₂, gradient_b), (D₁, gradient_a), (D₁D₂, hessian))
            @test maximum(abs.(Float64.(got) .- want)) ≤ tolerance * maximum(abs.(want))
        end
    end
end

@testset "the model's guards" begin
    nn = NeuralNetwork(
        LagrangianNeuralNetwork(2; width = 2, nhidden = 0, activation = square),
        Float64)
    discrete = NeuralNetwork(DiscreteLagrangianNetwork(1; width = 4), Float64)
    for Δt in (0, -0.1, NaN, Inf)
        message = "the timestep must be positive and finite, got $Δt"
        @test_throws ArgumentError(message) DiscreteLagrangian(nn, Δt)
    end

    @test_throws ArgumentError("a continuous Lagrangian needs a timestep: build `DiscreteLagrangian(nn, Δt)`") DiscreteLagrangian(
        nn)
    @test DiscreteLagrangian(discrete) isa DiscreteLagrangian

    # A timestep is used in the parameters' element type, and one that the conversion makes zero or
    # infinite is not the positive, finite timestep that was asked for.
    narrow = NeuralNetwork(
        LagrangianNeuralNetwork(2; width = 2, nhidden = 0, activation = square),
        Float32)
    for Δt in (1.0e-46, 1.0e40)
        @test_throws ArgumentError DiscreteLagrangian(narrow, Δt)
    end
end

@testset "a bare discrete network is the model it already is ($T)" for T in REAL_ELTYPES
    nn = quadratic_discrete(T)
    q₀ = T[0.3]
    q₁ = T[0.28]

    bare = variational_rollout(nn, q₀, q₁, 3)
    @test size(bare) == (1, 4)
    @test eltype(bare) === T
    @test bare[:, 3] ≈ -(q₀ .+ 2q₁)
    @test bare == variational_rollout(DiscreteLagrangian(nn), q₀, q₁, 3)
end

# ---------------------------------------------------------------- the two-position rollout

@testset "reference: the harmonic oscillator's closed form ($T)" for T in REAL_ELTYPES
    nn = oscillator(T)
    Δt = T(1 // 10)
    model = DiscreteLagrangian(nn, 1 // 10)
    q₀ = T[1]
    q₁ = T[0.99]
    Q = variational_rollout(model, q₀, q₁, 50)

    δ = Float64(Δt)
    θ = 2 * atan(δ / 2)
    a, b = Float64(q₀[1]), Float64(q₁[1])
    closed(k) = a * cos(k * θ) + (b - a * cos(θ)) * sin(k * θ) / sin(θ)

    @test size(Q) == (1, 51)
    @test eltype(Q) === T
    @test Q[:, 1] == q₀
    @test Q[:, 2] == q₁
    tolerance = T === Float64 ? 1.0e-11 : 1.0e-4
    @test maximum(abs.(Float64.(vec(Q)) .- [closed(k) for k in 0:50])) ≤ tolerance
end

@testset "one Newton step on a linear equation ($T)" for T in REAL_ELTYPES
    δ = T(1 // 10)
    S = T[0.2 0.7; -0.4 0.1]
    D₁(a, b) = -(b - a) / δ + S * b
    D₂(a, b) = (b - a) / δ + transpose(S) * a
    D₁D₂(a, b) = S - Matrix(I, 2, 2) / δ
    plain = DiscreteLagrangian(D₁, D₂, D₁D₂)

    nn = coupled(T)
    γ = T(3 // 10)
    midpoint = DiscreteLagrangian(
        (a, b) -> coupled_midpoint(a, b, δ, γ)[1],
        (a, b) -> coupled_midpoint(a, b, δ, γ)[2],
        (a, b) -> coupled_midpoint(a, b, δ, γ)[3])
    network = DiscreteLagrangian(nn, 1 // 10)

    q₀ = T[0.4, -0.3]
    q₁ = T[0.35, -0.25]
    Qplain = variational_rollout(plain, q₀, q₁, 20; maxiter = 1)
    @test size(Qplain) == (2, 21)
    @test Qplain isa Matrix{T}

    # The same trajectory from the momentum of the pair, one Newton step at a time. `S` is not
    # symmetric, so a Jacobian taken transposed lands on another `q₁` and leaves another trajectory.
    Qmomentum = variational_rollout(
        plain, q₀, -D₁(q₀, q₁), 20; momentum = true, maxiter = 1)
    @test size(Qmomentum) == (2, 21)
    @test Qmomentum isa Matrix{T}
    tolerance = T === Float64 ? 1.0e-11 : 1.0e-4
    @test maximum(abs.(Qmomentum .- Qplain)) ≤ tolerance

    Qnetwork = variational_rollout(network, q₀, q₁, 20; maxiter = 1)
    Qclosed = variational_rollout(midpoint, q₀, q₁, 20; maxiter = 1)
    @test size(Qnetwork) == (2, 21)
    @test maximum(abs.(Float64.(Qnetwork) .- Float64.(Qclosed))) ≤ tolerance
end

@testset "the Newton guess is the extrapolation of the last two positions ($T)" for T in REAL_ELTYPES
    δ = T(1 // 10)
    guesses = Vector{T}[]
    # The residual of this model is not its root at the guess, and `D₁L_d` records the first `c` it
    # is asked for, which is the position the solve starts from.
    D₁(a, b) = (isempty(guesses) && push!(guesses, copy(b)); b / δ - a)
    D₂(a, b) = a / δ - b
    D₁D₂(a, b) = Matrix{T}(I, 1, 1) / δ
    model = DiscreteLagrangian(D₁, D₂, D₁D₂)

    q₀ = T[1]
    q₁ = T[9 // 10]
    Q = variational_rollout(model, q₀, q₁, 2)
    @test size(Q) == (1, 3)
    @test length(guesses) == 1
    @test guesses[1] == 2q₁ - q₀
end

@testset "the counts ($T)" for T in REAL_ELTYPES
    δ = T(1 // 10)
    model = DiscreteLagrangian(oscillator_functions(δ)...)
    q₀ = T[1, 2]
    q₁ = T[3, 4]

    for (N, columns) in ((0, 1), (1, 2), (3, 4))
        Q = variational_rollout(model, q₀, q₁, N)
        @test size(Q) == (2, columns)
        @test Q[:, 1] == q₀
        if columns ≥ 2
            @test Q[:, 2] == q₁
        end
    end

    # A model of functions names no precision of its own: it is the promotion of the two positions'.
    @test eltype(variational_rollout(model, Float32[1, 2], Float64[3, 4], 2)) === Float64
    @test eltype(variational_rollout(model, Float32[1, 2], Float32[3, 4], 2)) === Float32
end

@testset "a solve that does not converge ($T)" for T in REAL_ELTYPES
    n = 2
    model = DiscreteLagrangian((a, b) -> ones(n), (a, b) -> ones(n),
        (a, b) -> Matrix{T}(I, n, n))
    error = try
        variational_rollout(model, T[1, 0], T[0, 1], 3)
    catch e
        e
    end
    @test error isa ErrorException
    @test occursin("q_2", error.msg)
    @test !occursin("q_3", error.msg)
end

@testset "a non-finite residual ($T)" for T in REAL_ELTYPES
    δ = T(1 // 10)
    D₁, D₂, D₁D₂ = oscillator_functions(δ)
    calls = Ref(0)
    counted(a, b) = (calls[] += 1; D₁D₂(a, b))
    model = DiscreteLagrangian(D₁, D₂, counted)

    error = try
        variational_rollout(model, T[NaN], T[1], 3)
    catch e
        e
    end
    @test error isa ErrorException
    @test occursin("q_2", error.msg)
    @test calls[] == 0
end

@testset "the other edges of the rollout" begin
    δ = 1 // 10
    model = DiscreteLagrangian(oscillator_functions(0.1)...)
    @test_throws ArgumentError("N must be non-negative, got -1") variational_rollout(
        model, [1.0], [0.99], -1)
    @test_throws ArgumentError("maxiter must be at least one, got 0") variational_rollout(
        model, [1.0], [0.99], 3; maxiter = 0)
    for tol in (0.0, -1.0)
        @test_throws ArgumentError("tol must be positive, got $tol") variational_rollout(
            model, [1.0], [0.99], 3; tol)
    end
    @test_throws DimensionMismatch("q₀ has 2 entries and q₁ has 1") variational_rollout(
        model, [1.0, 0.5], [0.99], 3)

    nn = NeuralNetwork(
        LagrangianNeuralNetwork(2; width = 2, nhidden = 0, activation = square),
        Float64)
    @test_throws DimensionMismatch("the positions have 2 entries, and the model takes 1") variational_rollout(
        DiscreteLagrangian(nn, δ), [1.0, 0.5], [0.99, 0.95], 3)
    @test_throws ArgumentError("a continuous Lagrangian needs a timestep: build `DiscreteLagrangian(nn, Δt)`") variational_rollout(
        nn, [1.0], [0.99], 3)

    # A network keeps its own precision when the positions are of another, so the two rollouts are
    # the same arithmetic and agree exactly.
    narrow = DiscreteLagrangian(oscillator(Float32), δ)
    Q = variational_rollout(narrow, [1.0], [0.99], 3)
    @test eltype(Q) === Float32
    @test Q == variational_rollout(narrow, Float32[1], Float32[0.99], 3)

    singular = DiscreteLagrangian((a, b) -> ones(1), (a, b) -> ones(1),
        (a, b) -> zeros(1, 1))
    @test_throws SingularException variational_rollout(singular, [1.0], [0.5], 3)
end

# ---------------------------------------------------------------- the momentum rollout

@testset "reference: the harmonic oscillator's first step ($T)" for T in REAL_ELTYPES
    nn = oscillator(T)
    Δt = T(1 // 10)
    model = DiscreteLagrangian(nn, 1 // 10)
    q₀ = T[1]
    p₀ = T[0.5]
    Q = variational_rollout(model, q₀, p₀, 50; momentum = true)

    δ = Float64(Δt)
    θ = 2 * atan(δ / 2)
    first_position = (Float64(p₀[1]) + Float64(q₀[1]) * (1 / δ - δ / 4)) / (1 / δ + δ / 4)
    closed(k) = Float64(q₀[1]) * cos(k * θ) +
                (first_position - Float64(q₀[1]) * cos(θ)) * sin(k * θ) / sin(θ)

    @test size(Q) == (1, 51)
    @test eltype(Q) === T
    @test Q[:, 1] == q₀
    @test isapprox(Q[1, 2], first_position; rtol = T === Float64 ? 1.0e-11 : 1.0e-4)
    tolerance = T === Float64 ? 1.0e-11 : 1.0e-4
    @test maximum(abs.(Float64.(vec(Q)) .- [closed(k) for k in 0:50])) ≤ tolerance
end

@testset "the momentum edges ($T)" for T in REAL_ELTYPES
    δ = T(1 // 10)
    _, D₂, D₁D₂ = oscillator_functions(δ)
    model = DiscreteLagrangian((a, b) -> error("D₁ must not be called"), D₂, D₁D₂)
    Q = variational_rollout(model, T[1, 2], T[0.5, -0.5], 0; momentum = true)
    @test size(Q) == (2, 1)
    @test Q[:, 1] == T[1, 2]

    n = 2
    failing = DiscreteLagrangian((a, b) -> ones(n), (a, b) -> ones(n),
        (a, b) -> Matrix{T}(I, n, n))
    error = try
        variational_rollout(failing, T[1, 0], T[0, 1], 3; momentum = true)
    catch e
        e
    end
    @test error isa ErrorException
    @test occursin("q_1", error.msg)
end

@testset "the momentum is solved in the parameters' type ($T)" for T in REAL_ELTYPES
    # A momentum of a strictly wider floating-point type: the residual's `D₁ + p` would promote the
    # whole solve into that type, where `T` is the parameters'.
    wide = T === Float32 ? Float64 : BigFloat
    model = DiscreteLagrangian(coupled(T), 1 // 10)
    q₀ = T[3 // 10, -1 // 5]
    p₀ = wide[1 // 4, -1 // 10]

    step = _solve_momentum(model, q₀, p₀, sqrt(eps(T)), 20)
    @test eltype(step) === T

    # The two rollouts do the same arithmetic once the momentum is converted, so they agree exactly.
    Q = variational_rollout(model, q₀, p₀, 5; momentum = true)
    @test eltype(Q) === T
    @test Q == variational_rollout(model, q₀, T.(p₀), 5; momentum = true)
end
