using Aqua
using ExplicitImports
using LagrangianNeuralNetworks
using Test

# `Aqua.test_all` opens a `@testset` per check and none around them, so on its own the first failing
# check would end the file; the `@testset` here is that parent.
@testset "Aqua" begin
    Aqua.test_all(LagrangianNeuralNetworks)
end

# A `using` or an `import` that nothing needs keeps a dependency reachable from the module file, so
# Aqua's `stale_deps` above would see a package in use that nothing uses. The names reached in
# other packages that those packages do not declare public are listed below; each is deliberate.
test_explicit_imports(LagrangianNeuralNetworks;
    all_explicit_imports_are_public = false,
    all_qualified_accesses_are_public = false)

const NON_PUBLIC_QUALIFIED = (
    :dim,                                         # AbstractNeuralNetworks
    :Jacobian, :build_nn_function)                # SymbolicNeuralNetworks
const NON_PUBLIC_IMPORTS = (
    :AbstractExplicitLayer, :Architecture, :dim,  # AbstractNeuralNetworks
    :derivative)                                  # SymbolicNeuralNetworks

@testset "every other name reached in another package is public" begin
    @test check_all_qualified_accesses_are_public(LagrangianNeuralNetworks;
        ignore = NON_PUBLIC_QUALIFIED) === nothing
    @test check_all_explicit_imports_are_public(LagrangianNeuralNetworks;
        ignore = NON_PUBLIC_IMPORTS) === nothing
end
