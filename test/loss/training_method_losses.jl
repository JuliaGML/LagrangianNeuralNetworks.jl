using LagrangianNeuralNetworks
using LagrangianNeuralNetworks: _discrete_lagrangian_derivatives
using GeometricMachineLearning: CPU, Adam, Batch, DataLoader, NetworkLoss,
                                NetworkParameters, NeuralNetwork, TrainingOptimizer, params,
                                train!
using LinearAlgebra
using Test
using Zygote
import Random

include("../helpers/eltypes.jl")

Random.seed!(1234)

# ---------------------------------------------------------------- ambiguity with `NetworkLoss`

@doc raw"""
Each three-argument loss functor `(loss)(ps, input, output)` must not collide with
`AbstractNeuralNetworks`' own `(::NetworkLoss)(::NeuralNetwork, input, output)`. With an untyped
first argument the two are ambiguous -- same argument count, neither more specific -- and
`loss(nn, input, output)` is a `MethodError: ... is ambiguous` rather than a call. Typing the first
parameter `::NetworkParameters` excludes a `NeuralNetwork`, so the call resolves to upstream's
method, which itself forwards to `loss(nn.params, input, output)` -- the two must therefore return
the same number.
"""
function test_loss_of_a_network_matches_loss_of_its_parameters(T::DataType)
    lnn_arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1)
    nn = NeuralNetwork(lnn_arch, CPU(), T)
    loss = LNNLoss(lnn_arch)
    input, output = randn(T, 2, 5), randn(T, 1, 5)
    @test loss(nn, input, output) == loss(params(nn), input, output)
    @test loss(nn, input, output) isa T

    vm_arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1)
    nn = NeuralNetwork(vm_arch, CPU(), T)
    loss = VariationalMidpointLoss(vm_arch, 0.05)
    input, output = randn(T, 2, 5), randn(T, 1, 5)
    @test loss(nn, input, output) == loss(params(nn), input, output)
    @test loss(nn, input, output) isa T
end

@testset "a loss of a network is the loss of its parameters ($T)" for T in REAL_ELTYPES
    test_loss_of_a_network_matches_loss_of_its_parameters(T)
end

"""
A timestep of the other precision must not change the element type of the loss: a `Float64`
timestep must not widen the loss of a `Float32` network, and a `Float32` one must not narrow the
loss of a `Float64` network. The timestep is a constant of the method, not data, so the data's
element type is what decides the result's.
"""
function test_timestep_does_not_widen_the_element_type(T::DataType, n::Integer = 1)
    lagrangian = LagrangianNeuralNetwork(2n; width = 4, nhidden = 1)
    psl = params(NeuralNetwork(lagrangian, CPU(), T))
    for timestep in (0.05, 0.05f0)
        @test VariationalMidpointLoss(lagrangian, timestep)(
            psl, randn(T, 2n, 5), randn(T, n, 5)) isa T
    end
end

@testset "the timestep does not change the element type ($T)" for T in REAL_ELTYPES
    test_timestep_does_not_widen_the_element_type(T)
end

# ---------------------------------------------------------------- VariationalMidpointLoss

@doc raw"""
The slot derivatives must be the derivatives of the midpoint discrete Lagrangian.

`_discrete_lagrangian_derivatives` writes the chain rule out by hand, so it has to be checked
against a `Zygote` gradient of ``L_d`` itself — which is the thing it exists in order not to call.
"""
function test_discrete_lagrangian_derivatives(T::DataType, n::Integer; Δt = T(0.07))
    arch = LagrangianNeuralNetwork(2n; width = 4, nhidden = 1)
    nn = NeuralNetwork(arch, CPU(), T)
    loss = VariationalMidpointLoss(arch, Δt)

    qa = randn(T, n, 1)
    qb = randn(T, n, 1)
    D₁, D₂ = _discrete_lagrangian_derivatives(loss, params(nn), qa, qb)
    @test eltype(D₁) === eltype(D₂) === T

    Ld(a, b) = Δt * sum(nn(vcat((a + b) / 2, (b - a) / Δt), params(nn)))
    @test vec(D₁) ≈ Zygote.gradient(a -> Ld(a, qb), qa)[1]
    @test vec(D₂) ≈ Zygote.gradient(b -> Ld(qa, b), qb)[1]
end

@testset "the slot derivatives of the midpoint discrete Lagrangian ($T)" for T in REAL_ELTYPES
    for n in (1, 2, 3)
        test_discrete_lagrangian_derivatives(T, n)
    end
end

@doc raw"""
The loss must *be* the discrete Euler–Lagrange residual, normalised.

Compare it against the same quantity assembled from `Zygote` gradients of ``L_d`` itself, which is
the call `VariationalMidpointLoss` exists in order not to make. This asserts what the loss computes
without needing a discrete trajectory to be solved for first.
"""
function test_variational_loss_is_the_del_residual(
        T::DataType, n::Integer; Δt = T(0.05), batch_size = 4)
    arch = LagrangianNeuralNetwork(2n; width = 4, nhidden = 1)
    nn = NeuralNetwork(arch, CPU(), T)
    loss = VariationalMidpointLoss(arch, Δt)

    qₙ = randn(T, n, batch_size)
    qₙ₊₁ = randn(T, n, batch_size)
    qₙ₊₂ = randn(T, n, batch_size)

    Ld(a, b) = Δt * sum(nn(vcat((a + b) / 2, (b - a) / Δt), params(nn)))
    D₂ = reduce(hcat,
        [Zygote.gradient(b -> Ld(qₙ[:, i], b), qₙ₊₁[:, i])[1] for i in 1:batch_size])
    D₁ = reduce(hcat,
        [Zygote.gradient(a -> Ld(a, qₙ₊₂[:, i]), qₙ₊₁[:, i])[1] for i in 1:batch_size])

    value = loss(params(nn), vcat(qₙ, qₙ₊₁), qₙ₊₂)
    @test value isa T
    @test value ≈ norm(D₂ + D₁) / norm(vcat(D₂, D₁))
end

@testset "the loss is the discrete Euler–Lagrange residual ($T)" for T in REAL_ELTYPES
    for n in (1, 2, 3)
        test_variational_loss_is_the_del_residual(T, n)
    end
end

@doc raw"""
The loss must vanish on a trajectory that satisfies the discrete Euler–Lagrange equations.

At one degree of freedom the residual is scalar, so ``q_{n+2}`` is found by bracketing a sign change
and bisecting. Bisection cannot diverge, which Newton on this residual does: a single overshoot puts
the midpoint velocity deep into `tanh` saturation, where the gradient is constant to machine
precision and a numerical Jacobian is exactly zero.
"""
function variational_next_position(loss, ps, qₙ, qₙ₊₁)
    _, D₂ = _discrete_lagrangian_derivatives(loss, ps, qₙ, qₙ₊₁)
    residual(q) = only(first(_discrete_lagrangian_derivatives(
        loss, ps, qₙ₊₁, reshape([q], 1, 1))) + D₂)

    centre = only(qₙ₊₁)
    candidates = range(centre - 5, centre + 5; length = 1001)
    values = residual.(candidates)
    crossing = findfirst(i -> values[i] * values[i + 1] < 0, 1:(length(values) - 1))
    crossing === nothing && return nothing

    lo, hi = candidates[crossing], candidates[crossing + 1]
    for _ in 1:60
        mid = (lo + hi) / 2
        if residual(lo) * residual(mid) ≤ 0
            hi = mid
        else
            lo = mid
        end
    end
    reshape([(lo + hi) / 2], 1, 1)
end

"""
    a_discrete_trajectory(loss, ps, T; Δt, attempts)

A triple `(qₙ, qₙ₊₁, qₙ₊₂)` of element type `T` that solves the discrete Euler–Lagrange equations
of `loss`, or `nothing` if no starting pair out of `attempts` admits one.

Not every starting pair does. The discrete Euler–Lagrange equations of an arbitrary neural-network
Lagrangian are a root-finding problem that need not have a solution near the starting pair, so this
searches rather than assuming.
"""
function a_discrete_trajectory(loss, ps, T::DataType; Δt = T(0.05), attempts = 20)
    for _ in 1:attempts
        qₙ = randn(T, 1, 1)
        qₙ₊₁ = qₙ + Δt * randn(T, 1, 1)
        qₙ₊₂ = variational_next_position(loss, ps, qₙ, qₙ₊₁)
        qₙ₊₂ === nothing || return (qₙ, qₙ₊₁, qₙ₊₂)
    end
    nothing
end

function test_variational_loss_vanishes_on_a_discrete_trajectory(T::DataType; Δt = T(0.05))
    arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1)
    nn = NeuralNetwork(arch, CPU(), T)
    loss = VariationalMidpointLoss(arch, Δt)

    trajectory = a_discrete_trajectory(loss, params(nn), T; Δt = Δt)
    @test trajectory !== nothing
    trajectory === nothing && return
    qₙ, qₙ₊₁, qₙ₊₂ = trajectory

    value = loss(params(nn), vcat(qₙ, qₙ₊₁), qₙ₊₂)
    @test value isa T
    @test value < 1000 * eps(T)
    # A position that does not solve the discrete Euler–Lagrange equations must not pass.
    @test loss(params(nn), vcat(qₙ, qₙ₊₁), qₙ₊₂ .+ T(0.1)) > 1.0e-3
end

# The seed makes the trajectory of each precision the same in every run, whatever ran before it.
@testset "the loss vanishes on a discrete trajectory ($T)" for T in REAL_ELTYPES
    Random.seed!(1234)
    test_variational_loss_vanishes_on_a_discrete_trajectory(T)
end

function test_variational_loss_gradient(T::DataType, n::Integer = 1)
    arch = LagrangianNeuralNetwork(2n; width = 4, nhidden = 1)
    nn = NeuralNetwork(arch, CPU(), T)
    loss = VariationalMidpointLoss(arch, 0.05)
    input = randn(T, 2n, 6)
    output = randn(T, n, 6)

    @test typeof(loss) <: NetworkLoss
    @test loss(nn.model, params(nn), input, output) == loss(params(nn), input, output)

    dp = Zygote.gradient(ps -> loss(ps, input, output), params(nn))[1]
    @test typeof(dp) <: NetworkParameters
    @test keys(dp) == keys(params(nn))
    @test eltype(dp.L1.W) === T
    @test !all(iszero, dp.L1.W)
end

@testset "VariationalMidpointLoss is differentiable ($T)" for T in REAL_ELTYPES
    test_variational_loss_gradient(T)
end

@doc raw"""
Training must reduce the loss on a real trajectory.

The data are the harmonic oscillator, ``q(t) = \cos(t)``, sampled at a fixed step. Asking the
network for a Lagrangian whose discrete Euler–Lagrange equations that trajectory satisfies is the
intended use of this loss, and it does not depend on the network's own initial draw the way a
self-generated trajectory would.
"""
function test_variational_training(T::DataType; Δt = T(0.05), samples = 60, n_epochs = 20)
    # The shape of the tests above. The assertion is that training reduces the loss, which does
    # not depend on the width and depth, and a new shape is a fresh `Zygote` specialisation of the
    # whole loss.
    arch = LagrangianNeuralNetwork(2; width = 4, nhidden = 1)
    nn = NeuralNetwork(arch, CPU(), T)
    loss = VariationalMidpointLoss(arch, Δt)

    t = range(zero(T), Δt * (samples + 1); step = Δt)
    q = cos.(t)
    qₙ = reshape(q[1:samples], 1, samples)
    qₙ₊₁ = reshape(q[2:(samples + 1)], 1, samples)
    qₙ₊₂ = reshape(q[3:(samples + 2)], 1, samples)

    dl = DataLoader(vcat(qₙ, qₙ₊₁), qₙ₊₂; suppress_info = true)
    loss_array = train!(
        nn, TrainingOptimizer(nn.params; algorithm = Adam()), dl, Batch(10), n_epochs;
        loss, show_progress = false)
    @test eltype(loss_array) === T
    @test loss_array[end] < loss_array[1]
end

@testset "training with VariationalMidpointLoss reduces the loss ($T)" for T in REAL_ELTYPES
    test_variational_training(T)
end
