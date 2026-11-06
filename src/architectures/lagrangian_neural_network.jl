@doc raw"""
    softplus(x)

The activation ``\log(1 + e^x)``, with the element type of `x`, evaluated in the split form
``x + \log(1 + e^{-x})`` for ``x > 0`` and ``\log(1 + e^{x})`` otherwise. Neither branch
overflows: the positive one adds a term that decays to `0`, the negative one an exponent below
`1`. For a large positive `x` the value is `x` to the accuracy of `x`, and for a large negative
one ``e^x`` to the accuracy of `exp`. So `softplus(-Inf)` is `0`, `softplus(Inf)` is `Inf`, and
`softplus(0)` is `log 2`.

It is the default activation of [`LagrangianNeuralNetwork`](@ref). With `tanh`, `LNNLoss` does
not learn the harmonic oscillator: the solve against ``\nabla_{\dot{q}}\nabla_{\dot{q}}L``
amplifies every sample where that block is near zero. With softplus it does;
`scripts/reproduction/k2_harmonic_oscillator.jl` trains it on 16 seeds in each precision.

The derivative of `softplus` is the logistic ``\sigma`` and its second derivative
``\sigma(1 - \sigma)``, both bounded, so the compiled first and second derivatives of a network
are finite in `Float32` and in `Float64`. The branches agree at `0` to every order, so their
tie-break changes no derivative.

LogExpFunctions also exports a `softplus`; a module that loads both packages qualifies the name.
"""
softplus(x) = ifelse(x > zero(x), x + log1p(exp(-x)), log1p(exp(x)))

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
