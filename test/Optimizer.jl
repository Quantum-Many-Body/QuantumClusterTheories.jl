using QuantumClusterTheories.Optimizer
using Optim: Options, optimize

@testset "Maximum" begin
    f(x) = -(x[1]-1)^4 - (x[2]-2)^4
    result = optimize(f, [0.0, 0.0], NoisyNewton(), Options(extended_trace=true))
    @test isapprox(result.minimizer, [1.0, 2.0]; atol=2e-3)
end

@testset "Minimum" begin
    f(x) = (x[1]-1)^4 + (x[2]-2)^4
    result = optimize(f, [0.0, 0.0], NoisyNewton(), Options(extended_trace=true))
    @test isapprox(result.minimizer, [1.0, 2.0]; atol=2e-3)
end

@testset "Saddle" begin
    f(x) = (x[1]-1)^4 - (x[2]-2)^4
    result = optimize(f, [0.0, 0.0], NoisyNewton(), Options(extended_trace=true))
    @test isapprox(result.minimizer, [1.0, 2.0]; atol=2e-3)
end
