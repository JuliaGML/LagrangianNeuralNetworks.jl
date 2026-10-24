using LagrangianNeuralNetworks: _euler_lagrange_acceleration
using Test

include("../helpers/eltypes.jl")

@doc raw"""
The Euler–Lagrange solve must give the pendulum's acceleration from the pendulum's Lagrangian.

For ``L(q, \dot{q}) = \tfrac{m}{2}\dot{q}^2 + \cos q`` the gradient with respect to the input
``(q, \dot{q})`` is ``(-\sin q, m\dot{q})`` and the Hessian is ``\mathrm{diag}(-\cos q, m)``, so the
Euler–Lagrange equation ``m\ddot{q} = -\sin q`` gives ``\ddot{q} = -\sin q / m``. A network cannot
represent this Lagrangian exactly, so the closed-form derivatives go to the solve directly.

``m = 1`` alone does not see a mass block that is not inverted; ``m = 2`` does. A flipped sign of
``\nabla_qL`` fails at both.
"""
function pendulum_lagrangian_derivatives(q::T, q̇::T, m::T) where {T <: Real}
    g = [-sin(q), m * q̇]
    H = [-cos(q) zero(T); zero(T) m]
    g, H
end

@testset "pendulum: q̈ = -sin q / m ($T, m = $m)" for T in REAL_ELTYPES, m in (1, 2)

    for q in range(-T(3), T(3); length = 7), q̇ in (-one(T), T(0.5), T(2))

        g, H = pendulum_lagrangian_derivatives(q, q̇, T(m))
        acceleration = _euler_lagrange_acceleration(H, g, [q̇])
        @test eltype(acceleration) === T
        @test abs(only(acceleration) - (-sin(q) / m)) ≤ 10 * eps(T)
    end
end
