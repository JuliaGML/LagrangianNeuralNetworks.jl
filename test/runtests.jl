using SafeTestsets

const GROUPS = isempty(ARGS) ? ["core", "slow"] : ARGS

if "core" in GROUPS
    @safetestset "Aqua and ExplicitImports" include("quality/aqua.jl")
    @safetestset "A loss takes a NetworkParameters" include("loss/losses.jl")
end
if "slow" in GROUPS
    @safetestset "Lagrangian Neural Network" include("architectures/lagrangian_neural_network_tests.jl")
    @safetestset "LNN and variational midpoint losses" include("loss/training_method_losses.jl")
end
