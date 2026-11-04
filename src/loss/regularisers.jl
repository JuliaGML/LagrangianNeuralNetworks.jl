# The guards of the constructors and the functors. Each returns its argument or throws.

function _check_timestep(Δt::Real)
    Δt > 0 && isfinite(Δt) ||
        throw(ArgumentError("the timestep must be positive and finite, got $Δt"))
    Δt
end

function _check_value(value::Real)
    value > 0 && isfinite(value) ||
        throw(ArgumentError("the value of the scale condition must be positive and finite, got $value"))
    value
end

function _check_weight(weight::Real)
    weight ≥ 0 && isfinite(weight) ||
        throw(ArgumentError("a weight must be finite and non-negative, got $weight"))
    weight
end

function _check_base(base::AbstractVector, arch::LagrangianArchitecture)
    length(base) == dim(arch) ||
        throw(DimensionMismatch("the base point has $(length(base)) entries, and the architecture needs $(dim(arch))"))
    base
end

# `input` as a matrix of `2n` rows, one column per sample.
function _input_columns(input::AbstractArray, D::LagrangianDerivatives)
    size(input, 1) == D.dim ||
        throw(DimensionMismatch("the input has $(size(input, 1)) rows, and the architecture needs $(D.dim)"))
    reshape(input, D.dim, :)
end

"""
    _check_form(form)

Return `form` if it is one of the forms of [`MixedHessianRegulariser`](@ref), `:inverse`, `:tamed`
or `:sigmoid`, and throw an `ArgumentError` otherwise.
"""
function _check_form(form)
    form in (:inverse, :tamed, :sigmoid) ||
        throw(ArgumentError("form must be :inverse, :tamed or :sigmoid, got $(repr(form))"))
    form
end

@doc raw"""
    _mixed_hessian(H, Δt)

The matrix ``\Lambda = -\Delta{}t\,D_1D_2L_d`` of the midpoint discrete Lagrangian
``L_d(a, b) = \Delta{}t\,L((a + b)/2, (b - a)/\Delta{}t)``, from the ``2n\times{}2n`` Hessian `H` of
``L`` at the midpoint, with ``D_1D_2L_d[i, j] = \partial^2L_d/\partial{}a_i\partial{}b_j``.

By the chain rule, with `H` in the blocks ``H_{qq}``, ``H_{q\dot{q}}`` and ``H_{\dot{q}\dot{q}}``,
``H_{q\dot{q}}[i, j] = \partial^2L/\partial{}q_i\partial\dot{q}_j``,

```math
    \Lambda = H_{\dot{q}\dot{q}}
        - \frac{\Delta{}t}{2}\left(H_{q\dot{q}} - H_{q\dot{q}}^T\right)
        - \frac{\Delta{}t^2}{4}H_{qq},
```

which tends to the velocity Hessian ``H_{\dot{q}\dot{q}}`` as ``\Delta{}t \to 0``. `Δt` is
converted to the element type of `H`.
"""
function _mixed_hessian(H::AbstractMatrix, Δt)
    n = size(H, 1) ÷ 2
    Δt = eltype(H)(Δt)
    Hqq = H[1:n, 1:n]
    Hqv = H[1:n, (n + 1):(2n)]
    Hvv = H[(n + 1):(2n), (n + 1):(2n)]
    Hvv - (Δt / 2) * (Hqv - transpose(Hqv)) - (Δt^2 / 4) * Hqq
end

# `det(A)`, and `NaN` for an `A` with a `NaN` or an `Inf` entry, where `det` itself returns `NaN` or
# `±Inf`: a penalty of such a matrix is then `NaN` in every form.
_det(A::AbstractMatrix) = all(isfinite, A) ? det(A) : eltype(A)(NaN)

# `‖A⁻¹‖_F²`, `Inf` for a singular `A`, and `NaN` for an `A` with a `NaN` or an `Inf` entry; `inv`
# throws for both of these.
function _inverse_frobenius2(A::AbstractMatrix)
    d = _det(A)
    isnan(d) && return d
    iszero(d) ? eltype(A)(Inf) : sum(abs2, inv(A))
end

"""
    _regularity_penalty(Λ, form)

The penalty of one sample of [`MixedHessianRegulariser`](@ref) on the matrix `Λ`, for `form`
`:inverse`, `:tamed` or `:sigmoid`, in the element type of `Λ`.
"""
function _regularity_penalty(Λ::AbstractMatrix, form::Symbol)
    T = eltype(Λ)
    if form === :inverse
        _inverse_frobenius2(Λ)
    elseif form === :tamed
        max(zero(T), 1 - 10 / _inverse_frobenius2(Λ))
    else
        1 - 1 / (1 + exp(-T(0.01) * _det(Λ)^2))
    end
end

@doc raw"""
    VelocityHessianRegulariser(arch)

The regularity penalty of a continuous Lagrangian, the continuous counterpart of the penalty (11) of
Offen & Ober-Blöbaum 2023 (see [`MixedHessianRegulariser`](@ref), whose ``\Lambda`` tends to the
velocity Hessian as ``\Delta{}t \to 0``): the mean over the samples of

```math
    \left\| (\nabla_{\dot{q}}\nabla_{\dot{q}}L)^{-1} \right\|_F^2 ,
```

which grows where the velocity Hessian is near singular, as it is for a null or a degenerate
Lagrangian. `arch` is a [`LagrangianArchitecture`](@ref).

Against the continuous penalty ``\sum\|(\partial^2L/\partial\dot{q}^2)^{-1}\|``, the norm is here
squared, taken in the Frobenius norm in place of the spectral norm (the two are equal for
``n = 1``, and the Frobenius norm is an upper bound otherwise), and averaged over the samples.

The result `r` is called as `r(ps, input)`, with `input` the points ``[q; \dot{q}]``: ``2n`` rows,
and any trailing dimensions, which count as samples. The value has the element type of `input`. A
singular velocity Hessian gives `Inf`, and an `input` of another row count than ``2n`` throws a
`DimensionMismatch`. Add it to a loss with [`RegularisedLoss`](@ref).
"""
struct VelocityHessianRegulariser{DT <: LagrangianDerivatives}
    derivatives::DT

    function VelocityHessianRegulariser(arch::LagrangianArchitecture)
        D = LagrangianDerivatives(arch)
        new{typeof(D)}(D)
    end
end

function (r::VelocityHessianRegulariser)(ps::NetworkParameters, input::AbstractArray)
    x = _input_columns(input, r.derivatives)
    n = r.derivatives.dim ÷ 2
    H = input_hessian(r.derivatives, x, ps)
    batch_size = size(x, 2)
    penalties = [_inverse_frobenius2(H[(n + 1):(2n), (n + 1):(2n), b])
                 for b in 1:batch_size]
    sum(penalties) / batch_size
end

@doc raw"""
    MixedHessianRegulariser(arch, Δt; form = :tamed)

The regularity penalty of the midpoint discrete Lagrangian
``L_d(q_0, q_1) = \Delta{}t\,L((q_0 + q_1)/2, (q_1 - q_0)/\Delta{}t)`` of a continuous Lagrangian:
the mean over the pairs ``[q_0; q_1]`` of a penalty of

```math
    \Lambda = -\Delta{}t\,\frac{\partial^2L_d}{\partial{}q_0\partial{}q_1},
```

see [`_mixed_hessian`](@ref). ``\Lambda`` tends to the velocity Hessian
``\nabla_{\dot{q}}\nabla_{\dot{q}}L`` as ``\Delta{}t \to 0``, so a threshold on it means the same as
one on the velocity Hessian. `arch` is a [`LagrangianArchitecture`](@ref), and `Δt` the timestep.

The three forms:

- `:inverse`: ``\|\Lambda^{-1}\|_F^2`` (Offen & Ober-Blöbaum 2023, (11)). The source, a paper on
  PDEs, penalises ``\|(D_1D_2L_d)^{-1}\|^2`` in the spectral norm; with
  ``\Lambda = -\Delta{}t\,D_1D_2L_d``, this is that penalty divided by ``\Delta{}t^2``, and in the
  Frobenius norm.
- `:tamed`: ``\mathrm{relu}(1 - 10/\|\Lambda^{-1}\|_F^2)`` (Offen & Ober-Blöbaum 2024, (34)), which
  is 0 wherever ``\|\Lambda^{-1}\|_F^2 \le 10``; the threshold 10 applies to the Frobenius norm of
  ``\Lambda^{-1}``.
- `:sigmoid`: ``1 - \sigma(0.01\det(\Lambda)^2)`` with ``\sigma(x) = 1/(1 + e^{-x})`` (Lishkova 2023,
  the degeneracy term added to the cost function (14), with ``m = 2``). The source uses
  ``d = \det D_2D_1L_d``; here
  ``\det\Lambda = (-\Delta{}t)^n\det D_1D_2L_d``, so that the three forms read the same matrix, and
  the square removes the sign.

A singular ``\Lambda`` gives `Inf` for `:inverse`, `1` for `:tamed` and `1/2` for `:sigmoid`.

The result `r` is called as `r(ps, input)`, with `input` the pairs ``[q_0; q_1]``: ``2n`` rows, and
any trailing dimensions, which count as samples. The value has the element type of `input`, and
`Δt` is converted to it. A timestep that is not positive and finite and an unknown `form` throw an
`ArgumentError`, and an `input` of another row count than ``2n`` a `DimensionMismatch`.
"""
struct MixedHessianRegulariser{DT <: LagrangianDerivatives, TT <: Real}
    derivatives::DT
    timestep::TT
    form::Symbol

    function MixedHessianRegulariser(
            arch::LagrangianArchitecture, Δt::Real; form::Symbol = :tamed)
        _check_timestep(Δt)
        _check_form(form)
        D = LagrangianDerivatives(arch)
        new{typeof(D), typeof(Δt)}(D, Δt, form)
    end
end

function (r::MixedHessianRegulariser)(ps::NetworkParameters, input::AbstractArray)
    x = _input_columns(input, r.derivatives)
    n = r.derivatives.dim ÷ 2
    Δt = eltype(x)(r.timestep)
    q₀ = x[1:n, :]
    q₁ = x[(n + 1):(2n), :]
    H = input_hessian(r.derivatives, vcat((q₀ + q₁) / 2, (q₁ - q₀) / Δt), ps)
    batch_size = size(x, 2)
    penalties = [_regularity_penalty(_mixed_hessian(H[:, :, b], Δt), r.form)
                 for b in 1:batch_size]
    sum(penalties) / batch_size
end

@doc raw"""
    ScaleCondition(arch; base = zeros(2n), value = 1)

The scale normalisation of a continuous Lagrangian (Offen 2025, (B.1)):

```math
    \left( \det\nabla_{\dot{q}}\nabla_{\dot{q}}L(x_b) - c \right)^2
```

at the point ``x_b = `` `base` ``= [q_b; \dot{q}_b]``, with ``c = `` `value`. It fixes the factor
by which a Lagrangian is determined. `arch` is a [`LagrangianArchitecture`](@ref) of ``2n`` inputs.

The result `r` is called as `r(ps, input)`; the value does not depend on `input`, whose row count
must still be ``2n``. `base` and `value` are converted to the element type of `input`, which the
value has. A `value` that is not positive and finite throws an `ArgumentError`, and a `base` of
another length than ``2n`` or an `input` of another row count a `DimensionMismatch`.
"""
struct ScaleCondition{DT <: LagrangianDerivatives, BT <: AbstractVector, VT <: Real}
    derivatives::DT
    base::BT
    value::VT

    function ScaleCondition(arch::LagrangianArchitecture;
            base::AbstractVector = zeros(dim(arch)), value::Real = 1)
        _check_value(value)
        _check_base(base, arch)
        D = LagrangianDerivatives(arch)
        new{typeof(D), typeof(base), typeof(value)}(D, base, value)
    end
end

function (r::ScaleCondition)(ps::NetworkParameters, input::AbstractArray)
    T = eltype(_input_columns(input, r.derivatives))
    n = r.derivatives.dim ÷ 2
    H = input_hessian(r.derivatives, reshape(T.(r.base), :, 1), ps)
    (_det(H[(n + 1):(2n), (n + 1):(2n), 1]) - T(r.value))^2
end

@doc raw"""
    DiscreteScaleCondition(arch, Δt; base = zeros(2n), value = 1)

The scale normalisation of the midpoint discrete Lagrangian of a continuous Lagrangian (Offen 2025,
(B.2)):

```math
    \left( \det\Lambda(x_b) - c \right)^2, \qquad
    \Lambda = -\Delta{}t\,\frac{\partial^2L_d}{\partial{}q_0\partial{}q_1},
```

with ``x_b = `` `base` the pair ``[q_0; q_1]``, ``\Lambda`` at its midpoint (see
[`_mixed_hessian`](@ref)), and ``c = `` `value`. The source has
``\det\partial^2L_d/\partial{}q_0\partial{}q_1``; the package uses
``\det\Lambda = (-\Delta{}t)^n\det\partial^2L_d/\partial{}q_0\partial{}q_1``, which tends to the
continuous ``\det\nabla_{\dot{q}}\nabla_{\dot{q}}L`` of [`ScaleCondition`](@ref) as
``\Delta{}t \to 0``.

The result `r` is called as `r(ps, input)`, as that of [`ScaleCondition`](@ref) is, with the same
conversions and refusals; a timestep that is not positive and finite throws an `ArgumentError`.
"""
struct DiscreteScaleCondition{
    DT <: LagrangianDerivatives, TT <: Real, BT <: AbstractVector, VT <: Real}
    derivatives::DT
    timestep::TT
    base::BT
    value::VT

    function DiscreteScaleCondition(arch::LagrangianArchitecture, Δt::Real;
            base::AbstractVector = zeros(dim(arch)), value::Real = 1)
        _check_timestep(Δt)
        _check_value(value)
        _check_base(base, arch)
        D = LagrangianDerivatives(arch)
        new{typeof(D), typeof(Δt), typeof(base), typeof(value)}(D, Δt, base, value)
    end
end

function (r::DiscreteScaleCondition)(ps::NetworkParameters, input::AbstractArray)
    T = eltype(_input_columns(input, r.derivatives))
    n = r.derivatives.dim ÷ 2
    Δt = T(r.timestep)
    q₀ = T.(r.base[1:n])
    q₁ = T.(r.base[(n + 1):(2n)])
    H = input_hessian(r.derivatives, reshape(vcat((q₀ + q₁) / 2, (q₁ - q₀) / Δt), :, 1), ps)
    (_det(_mixed_hessian(H[:, :, 1], Δt)) - T(r.value))^2
end

@doc raw"""
    RegularisedLoss(loss, terms...)

The loss `loss`, a `NetworkLoss`, plus weighted penalties: each term is a pair `weight => r`, and
the value is

```math
    \mathtt{loss}(ps, \mathtt{input}, \mathtt{output}) + \sum_k w_k\,r_k(ps, \mathtt{input}).
```

A penalty `r` is any callable `r(ps, input)`, such as [`VelocityHessianRegulariser`](@ref),
[`MixedHessianRegulariser`](@ref), [`ScaleCondition`](@ref) or [`DiscreteScaleCondition`](@ref).
Each weight is converted to the element type of `input`; with no terms, the value is that of
`loss`. A weight that is negative, `NaN` or `Inf` throws an `ArgumentError`. A weight of 0 is
allowed, and its penalty is not evaluated, so a penalty of `Inf` or `NaN` adds nothing there.

# Functor

```julia
loss(model, ps, input, output)
loss(ps, input, output) # equivalent to the above
```
"""
struct RegularisedLoss{LT <: NetworkLoss, TT <: Tuple} <: NetworkLoss
    loss::LT
    terms::TT

    function RegularisedLoss(loss::NetworkLoss, terms::Pair...)
        foreach(term -> _check_weight(first(term)), terms)
        new{typeof(loss), typeof(terms)}(loss, terms)
    end
end

# The sum of the weighted terms, `Σ weight·r(ps, input)`, in the element type `T`. A term of weight
# 0 is not evaluated, so that a penalty of `Inf` or `NaN` does not make it `NaN`.
_weighted_terms(::Tuple{}, ps, input, T) = zero(T)
function _weighted_terms(terms::Tuple, ps, input, T)
    weight, r = first(terms)
    rest = _weighted_terms(Base.tail(terms), ps, input, T)
    iszero(weight) ? rest : T(weight) * r(ps, input) + rest
end

function (loss::RegularisedLoss)(::Union{Chain, AbstractExplicitLayer},
        ps::NetworkParameters, input::AbstractArray, output::AbstractArray)
    loss(ps, input, output)
end

function (loss::RegularisedLoss)(
        ps::NetworkParameters, input::AbstractArray, output::AbstractArray)
    loss.loss(ps, input, output) + _weighted_terms(loss.terms, ps, input, eltype(input))
end
