# The benchmarks of the comparison and the integrator of their reference solutions.
#
# Each benchmark is a named tuple with
#
# - `name`, the name of its CSV file;
# - `n`, the number of positions;
# - `lagrangian`, the true Lagrangian as a function of the state `x = [q; q̇]`;
# - `acceleration`, the closed form `(q, q̇) ↦ q̈` of its Euler–Lagrange equations;
# - `lower` and `upper`, the corners of the box of `[q; q̇]` that the data and the measures cover;
# - `Δt`, the timestep of the position data and of the rollout;
# - `width`, the width of the networks trained on it;
# - `x0`, the state `[q; q̇]` the rollout's reference trajectory starts from.
#
# Every Lagrangian here has `L(0) = 0`, `∂_q̇L(0) = 0` and `det ∂²L/∂q̇²(0) = 1`, which are the
# normalisation and the scale condition that `compare.jl` imposes on the `euler_lagrange` method.
#
# The file loads no package, so that a check of the closed forms can include it beside Zygote alone.

# The coupling `α` of the Hénon–Heiles potential `½‖q‖² + α(q₁²q₂ − q₂³/3)`, the convention of
# Ober-Blöbaum & Offen 2023, §3.2.
const henon_heiles_alpha = 0.8

"""
    double_pendulum_acceleration(q, q̇)

The acceleration of the double pendulum of unit masses, lengths and gravity: the solution of
`M(q) q̈ = r(q, q̇)` with the mass matrix `M = [2 c; c 1]`, `c = cos(q₁ − q₂)`, and
`r = [-q̇₂² s − 2 sin q₁, q̇₁² s − sin q₂]`, `s = sin(q₁ − q₂)`, by Cramer's rule.
"""
function double_pendulum_acceleration(q, q̇)
    c, s = cos(q[1] - q[2]), sin(q[1] - q[2])
    r₁ = -q̇[2]^2 * s - 2 * sin(q[1])
    r₂ = q̇[1]^2 * s - sin(q[2])
    d = 2 - c^2
    [(r₁ - c * r₂) / d, (2 * r₂ - c * r₁) / d]
end

const BENCHMARKS = [
    (name = "harmonic_oscillator",
        n = 1,
        lagrangian = x -> (x[2]^2 - x[1]^2) / 2,
        acceleration = (q, q̇) -> [-q[1]],
        lower = [-1.0, -1.0], upper = [1.0, 1.0],
        Δt = 0.1, width = 4, x0 = [0.5, 0.0]),
    # Offen 2025, §5.1.
    (name = "coupled_oscillator",
        n = 2,
        lagrangian = x -> (x[3]^2 + x[4]^2) / 2 - (x[1]^2 + x[2]^2) / 2 + 0.1 * x[1] * x[2],
        acceleration = (q, q̇) -> [-q[1] + 0.1 * q[2], -q[2] + 0.1 * q[1]],
        lower = fill(-1.0, 4), upper = fill(1.0, 4),
        Δt = 0.1, width = 4, x0 = [0.5, -0.3, 0.0, 0.0]),
    # The form of `scripts/utilities/data_problem.jl`, whose potential `1 - cos q` vanishes at 0.
    (name = "pendulum",
        n = 1,
        lagrangian = x -> x[2]^2 / 2 - (1 - cos(x[1])),
        acceleration = (q, q̇) -> [-sin(q[1])],
        lower = [-1.2, -1.2], upper = [1.2, 1.2],
        Δt = 0.1, width = 4, x0 = [0.8, 0.0]),
    (name = "henon_heiles",
        n = 2,
        lagrangian = x -> (x[3]^2 + x[4]^2) / 2 - (x[1]^2 + x[2]^2) / 2 -
                          henon_heiles_alpha * (x[1]^2 * x[2] - x[2]^3 / 3),
        acceleration = (q, q̇) -> [
            -q[1] - 2 * henon_heiles_alpha * q[1] * q[2],
            -q[2] - henon_heiles_alpha * (q[1]^2 - q[2]^2)],
        lower = fill(-0.4, 4), upper = fill(0.4, 4),
        Δt = 0.1, width = 4, x0 = [0.3, -0.2, 0.0, 0.1]),
    # Unit masses, lengths and gravity, the potential shifted to `V(0) = 0`.
    (name = "double_pendulum",
        n = 2,
        lagrangian = x -> x[3]^2 + x[4]^2 / 2 + x[3] * x[4] * cos(x[1] - x[2]) -
                          2 * (1 - cos(x[1])) - (1 - cos(x[2])),
        acceleration = double_pendulum_acceleration,
        lower = fill(-1.0, 4), upper = fill(1.0, 4),
        Δt = 0.05, width = 4, x0 = [0.5, -0.5, 0.0, 0.0])
]

"""
    rk4(acceleration, q0, v0, h, N)

Integrate `q̈ = acceleration(q, q̇)` with the classical fourth-order Runge–Kutta method on the
first-order system of `(q, q̇)`, from `(q0, v0)` with `N` steps of size `h`.

Return `(q = Q, v = V)`, two `n × (N + 1)` matrices whose first columns are `q0` and `v0`.
"""
function rk4(acceleration, q0, v0, h, N)
    Q = zeros(length(q0), N + 1)
    V = zeros(length(v0), N + 1)
    Q[:, 1] .= q0
    V[:, 1] .= v0
    for k in 1:N
        q, v = Q[:, k], V[:, k]
        k1q, k1v = v, acceleration(q, v)
        k2q, k2v = v + h / 2 * k1v, acceleration(q + h / 2 * k1q, v + h / 2 * k1v)
        k3q, k3v = v + h / 2 * k2v, acceleration(q + h / 2 * k2q, v + h / 2 * k2v)
        k4q, k4v = v + h * k3v, acceleration(q + h * k3q, v + h * k3v)
        Q[:, k + 1] .= q + h / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
        V[:, k + 1] .= v + h / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    end
    (q = Q, v = V)
end
