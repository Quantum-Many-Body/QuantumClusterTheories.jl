using Test
using SafeTestsets

@safetestset "QuantumClusterTheories" begin
    @time @safetestset "Optimizer" begin include("Optimizer.jl") end
    @time @safetestset "Core" begin include("Core.jl") end
end