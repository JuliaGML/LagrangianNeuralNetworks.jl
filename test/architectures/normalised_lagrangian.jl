using LagrangianNeuralNetworks
using LagrangianNeuralNetworks: LagrangianDerivatives, input_gradient, SymbolicNeuralNetwork
using AbstractNeuralNetworks: AbstractNeuralNetworks, Chain, NeuralNetwork, params
using GeometricMachineLearning: CPU
using Test
using Zygote
import Random

include("../helpers/eltypes.jl")

Random.seed!(1234)

const ACTIVATIONS = (tanh, softplus, identity)

# The tolerance of the two identities, relative to `max(1, |NN(x_b)|)` or `max(1, ‖∂_q̇NN(x_b)‖)`:
# the wrapper subtracts a term of the wrapped network's own size, so the rounding scales with it.
identity_rtol(::Type{Float64}) = 1e-12
identity_rtol(::Type{Float32}) = 1e-5

# Every parameter array of `ps` multiplied by `s`, in place.
scale!(x::AbstractArray, s) = (x .*= s; x)
scale!(x, s) = (foreach(v -> scale!(v, s), values(x)); x)

# The parameters of each draw: the default initialiser's, and the same scaled by 3.
function parameter_draws(narch, T; ndraws = 5)
    draws = []
    for _ in 1:ndraws
        ps = params(NeuralNetwork(narch, CPU(), T))
        push!(draws, ps, scale!(deepcopy(ps), 3))
    end
    draws
end

# The wrapped network `NN` on the parameters of the wrapper, which nests them under `L1`.
wrapped(arch, x, ps) = Chain(arch)(x, ps.L1)

# `∂_q̇NN(x)` of the wrapped network, from `Zygote`.
function wrapped_momentum(arch, x, ps)
    n = length(x) ÷ 2
    Zygote.gradient(y -> sum(wrapped(arch, y, ps)), x)[1][(n + 1):(2n)]
end

@testset "the value at the base point ($T, n = $n, $act)" for T in REAL_ELTYPES,
    n in (1, 2),
    act in ACTIVATIONS
    arch = LagrangianNeuralNetwork(2n; width = 4, nhidden = 1, activation = act)
    base, momentum, value = randn(2n), randn(n), 0.37
    narch = NormalisedLagrangian(arch; base, value, momentum)
    @test narch isa LagrangianArchitecture
    @test AbstractNeuralNetworks.dim(narch) == 2n
    rtol = identity_rtol(T)
    xb, pb, cb = T.(base), T.(momentum), T(value)

    for ps in parameter_draws(narch, T)
        # the vector form
        L = Chain(narch)(xb, ps)
        NNb = only(wrapped(arch, xb, ps))
        @test eltype(L) === T
        @test size(L) == (1,)
        @test abs(only(L) - cb) ≤ rtol * max(1, abs(NNb))

        # the matrix form, with the base point as its first column; away from the base point the
        # value is the closed form of L̃, with `∂_q̇NN(x_b)` from `Zygote`
        X = hcat(xb, randn(T, 2n, 3))
        LX = Chain(narch)(X, ps)
        @test eltype(LX) === T
        @test size(LX) == (1, 4)
        @test abs(LX[1, 1] - cb) ≤ rtol * max(1, abs(NNb))
        gb = wrapped_momentum(arch, xb, ps)
        for j in 2:4
            x = X[:, j]
            NNx = only(wrapped(arch, x, ps))
            shift = sum((x[(n + 1):(2n)] - xb[(n + 1):(2n)]) .* (gb - pb))
            expected = NNx - NNb - shift + cb
            @test isapprox(LX[1, j], expected;
                rtol = sqrt(eps(T)), atol = sqrt(eps(T)) *
                                            max(1, abs(NNx), abs(NNb), abs(shift)))
            @test isapprox(only(Chain(narch)(x, ps)), LX[1, j]; rtol = 8 * eps(T),
                atol = 8 * eps(T) * max(1, abs(NNx), abs(NNb), abs(shift)))
        end
    end
end

@testset "the momentum at the base point ($T, n = $n, $act)" for T in REAL_ELTYPES,
    n in (1, 2),
    act in ACTIVATIONS
    arch = LagrangianNeuralNetwork(2n; width = 4, nhidden = 1, activation = act)
    base, momentum = randn(2n), randn(n)
    narch = NormalisedLagrangian(arch; base, value = 0.37, momentum)
    rtol = identity_rtol(T)
    xb, pb = T.(base), T.(momentum)

    for ps in parameter_draws(narch, T)
        g = Zygote.gradient(x -> sum(Chain(narch)(x, ps)), xb)[1]
        @test eltype(g) === T
        p̃ = g[(n + 1):(2n)]
        gNN = wrapped_momentum(arch, xb, ps)
        @test maximum(abs, p̃ - pb) ≤ rtol * max(1, maximum(abs, gNN))
    end
end

@testset "both identities in the traced expression (n = $n)" for n in (1, 2)
    arch = LagrangianNeuralNetwork(2n; width = 4, nhidden = 1, activation = softplus)
    base, momentum = randn(2n), randn(n)
    narch = NormalisedLagrangian(arch; base, value = 0.37, momentum)

    snn = SymbolicNeuralNetwork(narch)
    @test length(snn.input) == 2n

    # the gradient is compiled from the traced expression; the Hessian, which the losses below
    # compile, is not needed here
    D = LagrangianDerivatives(narch; hessian = false)
    for T in REAL_ELTYPES, ps in parameter_draws(narch, T; ndraws = 2)

        xb, pb = T.(base), T.(momentum)
        g = input_gradient(D, reshape(xb, 2n, 1), ps)
        @test size(g) == (2n, 1)
        @test eltype(g) === T
        gNN = wrapped_momentum(arch, xb, ps)
        @test maximum(abs, g[(n + 1):(2n), 1] - pb) ≤
              identity_rtol(T) * max(1, maximum(abs, gNN))
    end
end

@testset "a Float64 base point with Float32 parameters" begin
    n = 2
    arch = LagrangianNeuralNetwork(2n; width = 4)
    base, momentum, value = randn(2n), randn(n), 0.37
    narch = NormalisedLagrangian(arch; base, value, momentum)
    ps = params(NeuralNetwork(narch, CPU(), Float32))
    xb = Float32.(base)

    @test eltype(Chain(narch)(xb, ps)) === Float32
    @test eltype(Chain(narch)(randn(Float32, 2n, 3), ps)) === Float32
    @test eltype(Zygote.gradient(x -> sum(Chain(narch)(x, ps)), xb)[1]) === Float32
    @test abs(only(Chain(narch)(xb, ps)) - Float32(value)) ≤
          identity_rtol(Float32) * max(1, abs(only(wrapped(arch, xb, ps))))
end

@testset "value and momentum both zero ($T, n = $n)" for T in REAL_ELTYPES, n in (1, 2)

    arch = LagrangianNeuralNetwork(2n; width = 4)
    narch = NormalisedLagrangian(arch)
    rtol = identity_rtol(T)
    x0 = zeros(T, 2n)
    for ps in parameter_draws(narch, T)
        L = Chain(narch)(x0, ps)
        @test eltype(L) === T
        @test abs(only(L)) ≤ rtol * max(1, abs(only(wrapped(arch, x0, ps))))
        g = Zygote.gradient(x -> sum(Chain(narch)(x, ps)), x0)[1]
        @test eltype(g) === T
        @test maximum(abs, g[(n + 1):(2n)]) ≤
              rtol * max(1, maximum(abs, wrapped_momentum(arch, x0, ps)))
    end
end

# The exception that `f()` throws, or `nothing` when it throws none.
function thrown(f)
    try
        f()
    catch e
        return e
    end
    nothing
end

@testset "the guards of the construction" begin
    arch = LagrangianNeuralNetwork(4)

    err = thrown(() -> NormalisedLagrangian(arch; base = zeros(3)))
    @test err isa DimensionMismatch
    @test occursin(
        r"^NormalisedLagrangian: the base point has length 3, and the architecture takes 4 inputs$",
        err.msg)

    err = thrown(() -> NormalisedLagrangian(arch; base = zeros(5)))
    @test err isa DimensionMismatch
    @test occursin(
        r"^NormalisedLagrangian: the base point has length 5, and the architecture takes 4 inputs$",
        err.msg)

    err = thrown(() -> NormalisedLagrangian(arch; base = zeros(4), momentum = zeros(1)))
    @test err isa DimensionMismatch
    @test occursin(
        r"^NormalisedLagrangian: the momentum has length 1, and the architecture has 2 velocities$",
        err.msg)

    err = thrown(() -> NormalisedLagrangian(arch; momentum = zeros(3)))
    @test err isa DimensionMismatch
    @test occursin(
        r"^NormalisedLagrangian: the momentum has length 3, and the architecture has 2 velocities$",
        err.msg)

    # the right lengths, `2n` and `n`
    @test NormalisedLagrangian(arch; base = zeros(4), momentum = zeros(2)) isa
          NormalisedLagrangian

    # an odd input dimension has no split into `q` and `q̇`
    err = thrown(() -> NormalisedLagrangian(LagrangianNeuralNetwork(3)))
    @test err isa ArgumentError
    @test occursin(
        r"^NormalisedLagrangian: the architecture takes 3 inputs, and the input \[q; q̇\] has an even number of rows$",
        err.msg)

    softsign = x -> x / (1 + abs(x))
    for act in (softsign, sin)
        err = thrown(() -> NormalisedLagrangian(LagrangianNeuralNetwork(2; activation = act)))
        @test err isa ArgumentError
        name = act === sin ? "sin" : string(act)
        @test occursin(
            Regex("^NormalisedLagrangian: layer 1 has the activation " *
                  "\\Q$(name)\\E, and the reverse pass supports tanh, softplus and identity\$"),
            err.msg)
    end

    # a layer that is no `Dense`, here the layer of a wrapper wrapped again
    err = thrown(() -> NormalisedLagrangian(NormalisedLagrangian(arch)))
    @test err isa ArgumentError
    @test occursin(
        r"^NormalisedLagrangian: layer 1 is a NormalisedLagrangianLayer, and the reverse pass supports Dense and Linear layers$",
        err.msg)
end

# One wrapper for `n = 1` and its Euler–Lagrange loss, shared by the two testsets below so that the
# loss is compiled once.
const LOSS_ARCH = NormalisedLagrangian(LagrangianNeuralNetwork(2; width = 4);
    base = [0.3, -0.2], value = 0.37, momentum = [0.5])
const EL_LOSS = EulerLagrangeLoss(LOSS_ARCH)

@testset "three losses on the wrapper" begin
    losses = (LNNLoss(LOSS_ARCH), EL_LOSS, VariationalMidpointLoss(LOSS_ARCH, 0.05))
    for T in REAL_ELTYPES, loss in losses

        ps = params(NeuralNetwork(LOSS_ARCH, CPU(), T))
        value = loss(ps, randn(T, 2, 5), randn(T, 1, 5))
        @test value isa T
        @test isfinite(value)
    end
end

# Every parameter array of a nested parameter set, in the order of its keys.
leaves(x::AbstractArray) = [x]
leaves(x) = reduce(vcat, [leaves(v) for v in values(x)])

# The central difference of `f` in each entry of each parameter array of `ps`, with step
# `cbrt(eps(T))`, whose truncation and rounding errors are both of order `eps(T)^(2/3)`.
function finite_difference(f, ps, ::Type{T}) where {T}
    h = cbrt(eps(T))
    p = deepcopy(ps)
    map(leaves(p)) do A
        d = similar(A)
        for i in eachindex(A)
            a = A[i]
            A[i] = a + h
            fp = f(p)
            A[i] = a - h
            fm = f(p)
            A[i] = a
            d[i] = (fp - fm) / (2h)
        end
        d
    end
end

fd_rtol(::Type{Float64}) = 1e-6
fd_rtol(::Type{Float32}) = 1e-2

@testset "the parameter gradient through the wrapper ($T)" for T in REAL_ELTYPES
    narch = LOSS_ARCH
    ps = params(NeuralNetwork(narch, CPU(), T))
    input, output = randn(T, 2, 5), randn(T, 1, 5)

    # the Euler–Lagrange residual loss
    f = p -> EL_LOSS(p, input, output)
    g = leaves(Zygote.gradient(f, ps)[1])
    fd = finite_difference(f, ps, T)
    @test length(g) == length(fd) == length(leaves(ps))
    for (gi, fdi) in zip(g, fd)
        @test eltype(gi) === T
        @test all(isfinite, gi)
        @test isapprox(gi, fdi; rtol = fd_rtol(T))
    end

    # the forward pass itself, whose base-point terms depend on the parameters; the loss above does
    # not see them, because a constant and a term linear in q̇ leave the Euler–Lagrange residual
    # unchanged
    f = p -> sum(Chain(narch)(input, p))
    g = leaves(Zygote.gradient(f, ps)[1])
    fd = finite_difference(f, ps, T)
    for (gi, fdi) in zip(g, fd)
        @test eltype(gi) === T
        @test all(isfinite, gi)
        @test isapprox(gi, fdi; rtol = fd_rtol(T))
    end
end

@testset "the export" begin
    @test Base.isexported(LagrangianNeuralNetworks, :NormalisedLagrangian)
end
