using LagrangianNeuralNetworks
using LagrangianNeuralNetworks: _euler_lagrange_residual
using AbstractNeuralNetworks: NetworkLoss, NeuralNetwork, params
using GeometricMachineLearning: CPU, NetworkParameters
using LinearAlgebra
using Test
using Zygote
import Random

include("../helpers/eltypes.jl")

Random.seed!(1234)

"The error `f()` throws, or `nothing`."
function thrown(f)
    try
        f()
    catch e
        return e
    end
    return nothing
end

@doc raw"""
For ``L = \tfrac12\dot{q}^TM\dot{q} + q^TC\dot{q} - \tfrac12q^TKq`` the input gradient is
``(C\dot{q} - Kq, M\dot{q} + C^Tq)``, the input Hessian is ``[-K\ C; C^T\ M]``, and the
Euler–Lagrange equations ``M\ddot{q} + C^T\dot{q} = C\dot{q} - Kq`` give
``\ddot{q} = M^{-1}((C - C^T)\dot{q} - Kq)``. The residual of ``\ddot{q} + d`` is therefore
``Md``. `C` is not symmetric, so that the transpose of the mixed block does not cancel.
"""
function test_residual_against_closed_form(T::DataType, n::Integer)
    M = (A = rand(T, n, n); A' * A + n * I)
    K = (B = rand(T, n, n); B' * B + n * I)
    C = rand(T, n, n)
    q, q̇, d = rand(T, n), rand(T, n), rand(T, n)

    H = [-K C; C' M]
    g = vcat(C * q̇ - K * q, M * q̇ + C' * q)
    q̈ = M \ ((C - C') * q̇ - K * q)

    r = _euler_lagrange_residual(H, g, q̇, q̈ + d)
    @test eltype(r) === T
    @test isapprox(r, M * d; rtol = sqrt(eps(T)))
    r₀ = _euler_lagrange_residual(H, g, q̇, q̈)
    @test eltype(r₀) === T
    @test norm(r₀) ≤ sqrt(eps(T)) * norm(M * q̈)

    # The residual with `Cq̇` in place of `Cᵀq̇` differs, so the test above tells the two apart.
    without_transpose = M * (q̈ + d) + C * q̇ - g[1:n]
    @test !isapprox(without_transpose, M * d; rtol = sqrt(eps(T)))
end

@testset "the residual against a closed form ($T)" for T in REAL_ELTYPES
    for n in (2, 3)
        test_residual_against_closed_form(T, n)
    end
end

# One architecture per shape: each is a compile of the symbolic derivatives.
const ARCHS = Dict(n => LagrangianNeuralNetwork(2n; width = 4) for n in (1, 2))
const LOSSES = Dict(n => EulerLagrangeLoss(ARCHS[n]) for n in (1, 2))

@testset "the loss is the mean of the squared residuals ($T, n = $n)" for T in REAL_ELTYPES,
    n in (1, 2)

    nn = NeuralNetwork(ARCHS[n], CPU(), T)
    ps = params(nn)
    input, output = randn(T, 2n, 5), randn(T, n, 5)

    reference = sum(1:5) do b
        x = input[:, b]
        H = Zygote.hessian(y -> sum(nn(y, ps)), x)
        g = Zygote.gradient(y -> sum(nn(y, ps)), x)[1]
        sum(abs2, _euler_lagrange_residual(H, g, x[(n + 1):(2n)], output[:, b]))
    end / 5

    value = LOSSES[n](ps, input, output)
    @test value isa T
    @test isapprox(value, reference; rtol = sqrt(eps(T)))
end

@testset "a zero Lagrangian gives exactly zero ($T)" for T in REAL_ELTYPES
    ps = params(NeuralNetwork(ARCHS[2], CPU(), T))
    for k in keys(ps), a in values(getproperty(ps, k))

        fill!(a, zero(T))
    end
    value = LOSSES[2](ps, randn(T, 4, 6), randn(T, 2, 6))
    @test value isa T
    @test value == zero(T)
end

@testset "trailing dimensions ($T)" for T in REAL_ELTYPES
    ps = params(NeuralNetwork(ARCHS[2], CPU(), T))
    input, output = randn(T, 4, 3, 5), randn(T, 2, 3, 5)
    value = LOSSES[2](ps, input, output)
    @test value isa T
    @test isapprox(value, LOSSES[2](ps, reshape(input, 4, :), reshape(output, 2, :));
        rtol = 8 * eps(T))
end

@testset "an empty batch ($T)" for T in REAL_ELTYPES
    ps = params(NeuralNetwork(ARCHS[2], CPU(), T))
    err = thrown(() -> LOSSES[2](ps, zeros(T, 4, 0), zeros(T, 2, 0)))
    @test err isa ArgumentError
    @test occursin(
        r"^EulerLagrangeLoss: the batch is empty, and a mean over no samples is undefined$",
        err.msg)
end

@testset "a size mismatch ($T)" for T in REAL_ELTYPES
    ps = params(NeuralNetwork(ARCHS[2], CPU(), T))
    cases = (
        (randn(T, 4, 5), randn(T, 3, 5),
            r"^EulerLagrangeLoss: the output has 3 rows, and the network takes 2 velocities$"),
        (randn(T, 3, 5), randn(T, 2, 5),
            r"^EulerLagrangeLoss: the input has 3 rows, and the network takes 4$"),
        (randn(T, 4, 5), randn(T, 2, 4),
            r"^EulerLagrangeLoss: the input has the trailing size \(5,\), and the output \(4,\)$")
    )
    for (input, output, message) in cases
        err = thrown(() -> LOSSES[2](ps, input, output))
        @test err isa DimensionMismatch
        @test occursin(message, err.msg)
    end
end

@testset "the loss trains and keeps the element type ($T)" for T in REAL_ELTYPES
    loss = LOSSES[1]
    @test loss isa NetworkLoss
    nn = NeuralNetwork(ARCHS[1], CPU(), T)
    input, output = randn(T, 2, 8), randn(T, 1, 8)

    @test loss(params(nn), input, output) isa T
    @test loss(nn, input, output) == loss(params(nn), input, output)

    dp = Zygote.gradient(ps -> loss(ps, input, output), params(nn))[1]
    @test dp isa NetworkParameters
    @test keys(dp) == keys(params(nn))
    @test eltype(dp.L1.W) === T
    @test !all(iszero, dp.L1.W)

    @test_throws MethodError loss(params(params(nn)), input, output)
end
