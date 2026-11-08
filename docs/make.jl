using LagrangianNeuralNetworks
using Documenter
using DocumenterInterLinks

# GeometricMachineLearning trains the network on ReducedComplexityModeling's `Batch`, whose
# docstring is in that package's manual, so the pages link to it through its `stable/` inventory.
links = InterLinks(
    "ReducedComplexityModeling" => "https://juliarcm.github.io/ReducedComplexityModeling.jl/stable/"
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
        "Comparison" => "comparison.md",
        "Lagrangian Neural Network" => "architectures/lagrangian_neural_network.md"
    ],
    plugins = [links]
)

deploydocs(;
    repo = "github.com/JuliaGML/LagrangianNeuralNetworks.jl",
    devbranch = "main",
    push_preview = true
)
