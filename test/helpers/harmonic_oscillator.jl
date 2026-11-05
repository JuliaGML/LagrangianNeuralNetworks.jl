# The harmonic oscillator `L = (m q̇² − q²)/2` as a network, exactly: one hidden layer of two
# neurons with the activation `x²` and the identity as weights gives `(q², q̇²)`, and the output
# layer weighs them with `−1/2` and `m/2`. The including file imports `LagrangianNeuralNetwork`,
# `NeuralNetwork`, `CPU` and `params`.

square(x) = x^2

"""
    ho(T; m = 1)

The architecture and the parameters, of element type `T`, of the network that is
`L = (m q̇² − q²)/2`. Its input Hessian is `[-1 0; 0 m]` at every point.
"""
function ho(T::DataType; m = 1)
    arch = LagrangianNeuralNetwork(2; width = 2, nhidden = 0, activation = square)
    ps = params(NeuralNetwork(arch, CPU(), T))
    ps.L1.W .= [1 0; 0 1]
    ps.L1.b .= 0
    ps.L2.W .= [-1 / 2 m / 2]
    arch, ps
end
