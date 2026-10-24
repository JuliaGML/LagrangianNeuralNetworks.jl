using LagrangianNeuralNetworks
using Documenter
using DocumenterInterLinks

# The training infrastructure — `DataLoader`, `Batch`, `train!` — is GeometricMachineLearning's,
# and the pages link to it through that package's `latest/` inventory, which its site builds from
# its `main` branch.
links = InterLinks(
    "GeometricMachineLearning" => "https://juliagni.github.io/GeometricMachineLearning.jl/latest/"
)

makedocs(;
    modules = [LagrangianNeuralNetworks],
    sitename = "LagrangianNeuralNetworks.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        repolink = "https://github.com/JuliaGML/LagrangianNeuralNetworks.jl",
        canonical = "https://juliagml.github.io/LagrangianNeuralNetworks.jl"
    ),
    pages = [
        "Home" => "index.md",
        "Lagrangian Neural Network" => "architectures/lagrangian_neural_network.md"
    ],
    plugins = [links]
)

deploydocs(;
    repo = "github.com/JuliaGML/LagrangianNeuralNetworks.jl",
    devbranch = "main",
    push_preview = true
)
