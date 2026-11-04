@doc raw"""
    softplus(x)

The activation ``\log(1 + e^x)``, with the element type of `x`.

It is the default activation of [`LagrangianNeuralNetwork`](@ref). With `tanh`, `LNNLoss` does
not learn the harmonic oscillator: the solve against ``\nabla_{\dot{q}}\nabla_{\dot{q}}L``
amplifies every sample where that block is near zero. With softplus it does;
`scripts/reproduction/k2_harmonic_oscillator.jl` trains it on 16 seeds in each precision.

`exp` overflows, so `softplus(x)` is `Inf` above about 709 in `Float64` and about 88 in `Float32`.
There is no guard. The compiled input derivatives of a network are then `NaN`, and so is every
loss built on them. For negative `x`, `1 + exp(x)` rounds: the value loses relative accuracy, and
it is `0` below about −37 in `Float64` and about −17 in `Float32`.

LogExpFunctions also exports a `softplus`; a module that loads both packages qualifies the name.
"""
softplus(x) = log(1 + exp(x))

"""
    LagrangianNeuralNetwork(dimin; width = dimin, nhidden = 1, activation = softplus)

A Lagrangian ``L(q, \\dot{q})`` as a network of `dimin` ``= 2n`` inputs and a scalar output: a
`Dense` layer of `width` neurons, `nhidden` more of them, and a bias-free `Linear` output layer.
The hidden layers apply `activation`, by default [`softplus`](@ref).
"""
struct LagrangianNeuralNetwork{AT} <: LagrangianArchitecture
    dimin::Int
    width::Int
    nhidden::Int
    act::AT

    function LagrangianNeuralNetwork(dimin; width = dimin, nhidden = 1, activation = softplus)
        new{typeof(activation)}(dimin, width, nhidden, activation)
    end
end

@inline AbstractNeuralNetworks.dim(arch::LagrangianNeuralNetwork) = arch.dimin

function Chain(nn::LagrangianNeuralNetwork)
    inner_layers = Tuple(
        [Dense(nn.width, nn.width, nn.act) for _ in 1:nn.nhidden]
    )

    Chain(
        Dense(nn.dimin, nn.width, nn.act),
        inner_layers...,
        Linear(nn.width, 1; use_bias = false)
    )
end
