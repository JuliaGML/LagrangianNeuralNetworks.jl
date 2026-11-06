@doc raw"""
    NormalisedLagrangian(arch::LagrangianArchitecture; base = zeros(2n), value = 0, momentum = zeros(n))

The Lagrangian of `arch` with its value and its momentum fixed at one base point (Offen 2025,
Prop. 3.1), where ``2n`` is `dim(arch)`:

```math
    \tilde{L}(x) = L(x) - L(x_b) - (\dot{q} - \dot{q}_b)^T\left(\nabla_{\dot{q}}L(x_b) - p_b\right) + c_b,
```

with ``x = (q, \dot{q})``, ``L`` the network of `arch`, ``x_b`` = `base`, ``\dot{q}_b`` its last
``n`` entries, ``c_b`` = `value` and ``p_b`` = `momentum`. Then ``\tilde{L}(x_b) = c_b`` and
``\nabla_{\dot{q}}\tilde{L}(x_b) = p_b`` for every value of the parameters, which removes the
constant and the linear gauge of a Lagrangian. With `value` and `momentum` both zero, the scale is
not fixed, and is left to a condition of its own.

The network takes the same input as that of `arch` and has the same parameters, nested under the
key `L1`. Every loss of the package takes a `NormalisedLagrangian` as it takes `arch`.

``\nabla_{\dot{q}}L(x_b)`` is computed in the forward pass, by a reverse pass over the layers of
`Chain(arch)`, so it is part of the expression that `SymbolicNeuralNetworks` traces and it depends
on the parameters. That reverse pass supports `Dense` and `Linear` layers with the activations
`tanh`, [`softplus`](@ref) and `identity`.

`base` and `momentum` are copied at construction. `base`, `value` and `momentum` are converted to
the element type of the parameters. Where the network of `arch` is not finite at the base point,
the network of the wrapper is not finite anywhere.

A `base` or a `momentum` that is no `AbstractVector` throws a `TypeError`. A `base` of a length
other than ``2n`` or a `momentum` of a length other than ``n`` throws a `DimensionMismatch`. An odd
`dim(arch)`, a layer of `Chain(arch)` that is no `Dense`, and an activation other than the three
above throw an `ArgumentError`.
"""
struct NormalisedLagrangian{AT <: LagrangianArchitecture, BT <: AbstractVector, VT <: Real,
    PT <: AbstractVector} <: LagrangianArchitecture
    arch::AT
    base::BT
    value::VT
    momentum::PT

    function NormalisedLagrangian(arch::LagrangianArchitecture;
            base::AbstractVector = zeros(dim(arch)), value::Real = 0,
            momentum::AbstractVector = zeros(dim(arch) ÷ 2))
        d = dim(arch)
        iseven(d) ||
            throw(ArgumentError("NormalisedLagrangian: the architecture takes $d inputs, and the input [q; q̇] has an even number of rows"))
        n = d ÷ 2
        length(base) == 2n ||
            throw(DimensionMismatch("NormalisedLagrangian: the base point has length $(length(base)), and the architecture takes $(2n) inputs"))
        length(momentum) == n ||
            throw(DimensionMismatch("NormalisedLagrangian: the momentum has length $(length(momentum)), and the architecture has $n velocities"))
        for (i, layer) in enumerate(Chain(arch).layers)
            layer isa Dense ||
                throw(ArgumentError("NormalisedLagrangian: layer $i is a $(nameof(typeof(layer))), and the reverse pass supports Dense and Linear layers"))
            layer.σ isa _ReversibleActivation ||
                throw(ArgumentError("NormalisedLagrangian: layer $i has the activation $(_activation_function(layer.σ)), and the reverse pass supports tanh, softplus and identity"))
        end
        b, p = copy(base), copy(momentum)
        new{typeof(arch), typeof(b), typeof(value), typeof(p)}(arch, b, value, p)
    end
end

@inline AbstractNeuralNetworks.dim(arch::NormalisedLagrangian) = dim(arch.arch)

function Chain(arch::NormalisedLagrangian)
    Chain(NormalisedLagrangianLayer{dim(arch)}(
        Chain(arch.arch), arch.base, arch.value, arch.momentum))
end

# The activations whose derivative the reverse pass knows. `Dense` stores `tanh` as a
# `TanhActivation`, any other function as a `GenericActivation` of it, and `Linear` has an
# `IdentityActivation`.
const _ReversibleActivation = Union{TanhActivation, IdentityActivation,
    GenericActivation{typeof(softplus)}, GenericActivation{typeof(identity)}}

_activation_function(σ::GenericActivation) = σ.σ
_activation_function(σ) = σ

# The derivative of an activation at the pre-activation `z`. That of softplus is written with
# `exp(-z)`, which overflows to `Inf` and gives `0` for large negative `z`, where
# `exp(z) / (1 + exp(z))` would give `NaN` for large positive `z`.
_activation_derivative(::TanhActivation, z) = 1 - tanh(z)^2
_activation_derivative(::GenericActivation{typeof(softplus)}, z) = 1 / (1 + exp(-z))
_activation_derivative(::GenericActivation{typeof(identity)}, z) = one(z)
_activation_derivative(::IdentityActivation, z) = one(z)

_preactivation(::Dense{M, N, true}, ps, a) where {M, N} = ps.W * a .+ ps.b
_preactivation(::Dense{M, N, false}, ps, a) where {M, N} = ps.W * a

# The value of a chain of `Dense` layers with a scalar output at the vector `a`, and its gradient
# with respect to `a`: the forward pass through the layers, then the reverse pass through their
# transposed weights and activation derivatives.
_value_and_gradient(::Tuple{}, a, ::Tuple{}) = (only(a), one.(a))
function _value_and_gradient(layers::Tuple, a, ps::Tuple)
    layer, p = first(layers), first(ps)
    z = _preactivation(layer, p, a)
    value, ∂ = _value_and_gradient(Base.tail(layers), layer.σ.(z), Base.tail(ps))
    value, transpose(p.W) * (_activation_derivative.((layer.σ,), z) .* ∂)
end

# The one layer of the `Chain` of a `NormalisedLagrangian`: it applies `chain` to an input of `M`
# rows and adds the normalisation at the base point. Its parameters are those of `chain`.
struct NormalisedLagrangianLayer{M, CT <: Chain, BT, VT, PT} <: AbstractExplicitLayer{M, 1}
    chain::CT
    base::BT
    value::VT
    momentum::PT

    function NormalisedLagrangianLayer{M}(chain::Chain, base, value, momentum) where {M}
        new{M, typeof(chain), typeof(base), typeof(value), typeof(momentum)}(
            chain, base, value, momentum)
    end
end

function AbstractNeuralNetworks.initialparameters(rng::AbstractRNG, init::Initializer,
        layer::NormalisedLagrangianLayer, backend::NeuralNetworkBackend, ::Type{T};
        kwargs...) where {T}
    params(initialparameters(rng, init, layer.chain, backend, T; kwargs...))
end

# The constant `c` of the normalisation in the element type of the parameter `w`. For a symbolic
# `w`, the call stays a term of the traced expression and converts `c` when the compiled code runs:
# a `Float64` constant in that expression would widen the result of a `Float32` network.
_like(c, w) = oftype(w, c)
@register_symbolic _like(c::Real, w)

function (layer::NormalisedLagrangianLayer{M})(x::AbstractVecOrMat, ps::NamedTuple) where {M}
    n = M ÷ 2
    w = first(first(values(ps)).W)
    xb = _like.(layer.base, (w,))
    cb = _like(layer.value, w)
    pb = _like.(layer.momentum, (w,))
    Lb, ∇Lb = _value_and_gradient(layer.chain.layers, xb, values(ps))
    q̇ = selectdim(x, 1, (n + 1):(2n))
    shift = transpose(∇Lb[(n + 1):(2n)] - pb) * (q̇ .- xb[(n + 1):(2n)])
    layer.chain(x, ps) .- shift .+ (cb - Lb)
end
