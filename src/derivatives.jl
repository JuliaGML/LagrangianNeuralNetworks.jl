@doc raw"""
    LagrangianDerivatives(arch::Architecture; hessian = true)

Compile the gradient and the Hessian of the network of `arch` with respect to its input, once.

`arch` is any architecture whose `Chain` has a scalar output and that has a method of
`AbstractNeuralNetworks.dim`, the input dimension ``2n``; it need not be a
[`LagrangianArchitecture`](@ref). Evaluate the result with [`input_gradient`](@ref) and
[`input_hessian`](@ref). With `hessian = false`, only the gradient is built and compiled, and
`input_hessian` throws an `ArgumentError`.

# Implementation

The gradient and the Hessian are built as *symbolic* expressions with
`SymbolicNeuralNetworks.Jacobian`, applied once and then again to its own result, and compiled with
`build_nn_function`. They are not taken with `Zygote` inside a loss: differentiating a loss that
itself contains a `Zygote.gradient` with respect to the network parameters fails with
`MethodError: no method matching getindex(::IdDict{Any, Any})`. This is the route
`hamiltonian_vector_field` of HamiltonianNeuralNetworks.jl takes, and for the same reason.

The compiled functions flatten a batch along the columns, ``(1, 2n\cdot{}B)`` for the gradient and
``(2n, 2n\cdot{}B)`` for the Hessian, so `input_gradient` and `input_hessian` reshape their output.
"""
struct LagrangianDerivatives{GT, HT}
    dim::Int
    gradient::GT
    hessian::HT
end

function LagrangianDerivatives(arch::Architecture; hessian::Bool = true)
    snn = SymbolicNeuralNetwork(arch)
    ∇L = derivative(SymbolicNeuralNetworks.Jacobian(snn))
    grad = SymbolicNeuralNetworks.build_nn_function(
        ∇L, snn.params, snn.input; inplace = false)
    hess = if hessian
        ∇∇L = derivative(SymbolicNeuralNetworks.Jacobian(∇L, snn))
        SymbolicNeuralNetworks.build_nn_function(
            ∇∇L, snn.params, snn.input; inplace = false)
    else
        nothing
    end
    LagrangianDerivatives(dim(arch), grad, hess)
end

# `x` as a matrix of `2n` rows, one column per sample, where `2n` is the input dimension of `D`; a
# `DimensionMismatch` when the first dimension of `x` is not `2n`.
function _input_matrix(D::LagrangianDerivatives, x::AbstractArray)
    size(x, 1) == D.dim ||
        throw(DimensionMismatch("the input has $(size(x, 1)) rows, and the network takes $(D.dim)"))
    reshape(x, D.dim, :)
end

@doc raw"""
    input_gradient(D::LagrangianDerivatives, x, ps)

The gradient of the network with respect to its input, ``\nabla_xL``, at each column of `x`, for
the parameters `ps`.

For `x` of size ``(2n, B)`` the result has size ``(2n, B)``. Trailing dimensions are flattened on
the way in and restored on the way out: `x` of size ``(2n, s, B)`` gives ``(2n, s, B)``. An `x`
whose first dimension is not ``2n`` throws a `DimensionMismatch`.
"""
function input_gradient(D::LagrangianDerivatives, x::AbstractArray, ps)
    reshape(D.gradient(_input_matrix(D, x), ps), D.dim, size(x)[2:end]...)
end

@doc raw"""
    input_hessian(D::LagrangianDerivatives, x, ps)

The Hessian of the network with respect to its input, ``\nabla_x\nabla_xL``, at each column of
`x`, for the parameters `ps`.

For `x` of size ``(2n, B)`` the result has size ``(2n, 2n, B)``, and slice `b` is the Hessian at
column `b`. Trailing dimensions are flattened on the way in and restored on the way out: `x` of
size ``(2n, s, B)`` gives ``(2n, 2n, s, B)``. An `x` whose first dimension is not ``2n`` throws a
`DimensionMismatch`, and `D` built with `hessian = false` an `ArgumentError`.
"""
function input_hessian(D::LagrangianDerivatives, x::AbstractArray, ps)
    reshape(D.hessian(_input_matrix(D, x), ps), D.dim, D.dim, size(x)[2:end]...)
end

function input_hessian(::LagrangianDerivatives{GT, Nothing}, ::AbstractArray, ps) where {GT}
    throw(ArgumentError("these derivatives were compiled with `hessian = false`, and hold no input Hessian"))
end
