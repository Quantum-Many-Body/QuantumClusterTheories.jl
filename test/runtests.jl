using Test
using SafeTestsets

@safetestset "QuantumClusterTheories" begin
    @time @safetestset "Optimizer" begin include("Optimizer.jl") end
    @time @safetestset "Core" begin include("Core.jl") end
    @time @safetestset "Transport" begin include("Transport.jl") end
end