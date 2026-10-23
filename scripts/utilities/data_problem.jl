using Zygote

#################################################################################################
# Lagrangian DictionaryHH

dict_problem_L = Dict(
    :pendulum => (x -> x[2]^2 / 2 - (1-cos(x[1])), 1),
)

###############################################################################
# get data of a problem for LNN (results is (data = (q,q̇), target = (qdotdot)))

function get_LNN_data(nameproblem, num = 10, qq̇min = -1.2, qq̇max = +1.2)

    #get the Hamiltonien corresponding to name_problem
    L, n_dim = dict_problem_L[nameproblem]

    #compute gradient and hessian
    ∇qL(x) = Zygote.gradient(x->L(x), x)[1][1:(length(x) ÷ 2)]
    ∇∇L(x) = Zygote.hessian(L, x)
    ∇q̇∇q̇L(x) = ∇∇L(x)[(1 + length(x) ÷ 2):end, (1 + length(x) ÷ 2):end]
    ∇q∇q̇L(x) = ∇∇L(x)[1:(length(x) ÷ 2), (1 + length(x) ÷ 2):end]

    # Solve the Euler-Lagrange equations for the acceleration. The transpose is load-bearing:
    # `∇q∇q̇L` is indexed `[i, j] = ∂²L/∂qᵢ∂q̇ⱼ` and the chain rule contracts the *first* index with
    # `q̇`. It vanishes for the pendulum, whose position and velocity do not couple, and does not for
    # a Lagrangian that couples them. `LagrangianNeuralNetworks.LNNLoss` solves the same equation.
    Qdotdot(q, q̇) = inv(∇q̇∇q̇L([q..., q̇...])) *
                    (∇qL([q..., q̇...]) - ∇q∇q̇L([q..., q̇...])' * q̇)

    #range in which the data should be in
    rang = range(qq̇min, stop = qq̇max, length = num)

    pre_data = collect(Iterators.product(fill(rang, 2*n_dim)...))
    data = [[[x[1:n_dim]...], [x[(1 + n_dim):(2 * n_dim)]...]] for x in pre_data]

    #compute target associate to data
    target = [Qdotdot(qq̇[1], qq̇[2]) for qq̇ in data]

    # `(input, output)` for `DataLoader`: `input` stacks q on q̇ and is `2 n_dim × N`, `output` is
    # the acceleration and is `n_dim × N`.
    (reduce(hcat, [vcat(qq̇[1], qq̇[2]) for qq̇ in vec(data)]), reduce(hcat, vec(target)))
end
