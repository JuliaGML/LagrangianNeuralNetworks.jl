using LagrangianNeuralNetworks
using Documenter
using DocumenterCitations
using DocumenterInterLinks

# GeometricMachineLearning trains the network on ReducedComplexityModeling's `Batch`, whose
# docstring is in that package's manual, so the pages link to it through its `stable/` inventory.
links = InterLinks(
    "ReducedComplexityModeling" => "https://juliarcm.github.io/ReducedComplexityModeling.jl/stable/"
)

# The sources each method section cites, listed on the References page.
bib = CitationBibliography(joinpath(@__DIR__, "src", "LagrangianNeuralNetworks.bib"))

makedocs(;
    modules = [LagrangianNeuralNetworks],
    sitename = "LagrangianNeuralNetworks.jl",
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", nothing) == "true",
        repolink = "https://github.com/JuliaGML/LagrangianNeuralNetworks.jl",
        canonical = "https://juliagml.github.io/LagrangianNeuralNetworks.jl",
        edit_link = "main"
    ),
    pages = [
        "Home" => "index.md",
        "Comparison" => "comparison.md",
        "Lagrangian Neural Network" => "architectures/lagrangian_neural_network.md",
        "Tutorial" => "tutorials/lagrangian_neural_network.md",
        "References" => "references.md"
    ],
    plugins = [bib, links]
)

deploydocs(;
    repo = "github.com/JuliaGML/LagrangianNeuralNetworks.jl",
    devbranch = "main",
    push_preview = true
)
