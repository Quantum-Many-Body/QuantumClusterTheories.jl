using ExactDiagonalization
using LinearAlgebra: I, dot, eigen, inv, tr
using QuantumClusterTheories
using QuantumLattices
using TightBindingApproximation
using TimerOutputs: TimerOutput
import CairoMakie as Makie
import Plots

@testset "Operators & Perturbation & Periodization" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (2, 2), ('P', 'P'))
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site in eachindex(lattice))
    @test operators(Fermionic(:TBA), lattice, hilbert) == [
        𝕔(1, 1, -1//2, [0.0, 0.0], [0.0, 0.0]), 𝕔(1, 1, 1//2, [0.0, 0.0], [0.0, 0.0]), 𝕔(2, 1, -1//2, [1.0, 0.0], [0.0, 0.0]), 𝕔(2, 1, 1//2, [1.0, 0.0], [0.0, 0.0]),
        𝕔(3, 1, -1//2, [0.0, 1.0], [0.0, 0.0]), 𝕔(3, 1, 1//2, [0.0, 1.0], [0.0, 0.0]), 𝕔(4, 1, -1//2, [1.0, 1.0], [0.0, 0.0]), 𝕔(4, 1, 1//2, [1.0, 1.0], [0.0, 0.0])
    ]
    @test operators(Fermionic(:BdG), lattice, hilbert) == [
        𝕔(1, 1, -1//2, [0.0, 0.0], [0.0, 0.0]), 𝕔(1, 1, 1//2, [0.0, 0.0], [0.0, 0.0]), 𝕔(2, 1, -1//2, [1.0, 0.0], [0.0, 0.0]), 𝕔(2, 1, 1//2, [1.0, 0.0], [0.0, 0.0]),
        𝕔(3, 1, -1//2, [0.0, 1.0], [0.0, 0.0]), 𝕔(3, 1, 1//2, [0.0, 1.0], [0.0, 0.0]), 𝕔(4, 1, -1//2, [1.0, 1.0], [0.0, 0.0]), 𝕔(4, 1, 1//2, [1.0, 1.0], [0.0, 0.0]),
        𝕔⁺(1, 1, -1//2, [0.0, 0.0], [0.0, 0.0]), 𝕔⁺(1, 1, 1//2, [0.0, 0.0], [0.0, 0.0]), 𝕔⁺(2, 1, -1//2, [1.0, 0.0], [0.0, 0.0]), 𝕔⁺(2, 1, 1//2, [1.0, 0.0], [0.0, 0.0]),
        𝕔⁺(3, 1, -1//2, [0.0, 1.0], [0.0, 0.0]), 𝕔⁺(3, 1, 1//2, [0.0, 1.0], [0.0, 0.0]), 𝕔⁺(4, 1, -1//2, [1.0, 1.0], [0.0, 0.0]), 𝕔⁺(4, 1, 1//2, [1.0, 1.0], [0.0, 0.0])
    ]

    t, U = Hopping(:t, 1.0, 1), Hubbard(:U, 0.0)
    @test quadratic((t, U)) == (t,)

    pert = Perturbation(lattice, hilbert, (t, U))
    @test kind(pert) == kind(typeof(pert)) == Fermionic(:TBA)
    @test pert() == [
        0.0 0.0 1.0 0.0 1.0 0.0 0.0 0.0;
        0.0 0.0 0.0 1.0 0.0 1.0 0.0 0.0;
        1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
        0.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0;
        1.0 0.0 0.0 0.0 0.0 0.0 1.0 0.0;
        0.0 1.0 0.0 0.0 0.0 0.0 0.0 1.0;
        0.0 0.0 1.0 0.0 1.0 0.0 0.0 0.0;
        0.0 0.0 0.0 1.0 0.0 1.0 0.0 0.0
    ]

    opsₗ = operators(Fermionic(:TBA), lattice, hilbert)
    opsᵤ = operators(Fermionic(:TBA), unitcell, hilbert)
    periodization = Periodization(opsₗ, opsᵤ, unitcell.vectors)
    @test periodization.coordinates == reduce(vcat, [[lattice[i], lattice[i]] for i in eachindex(lattice)])
    @test periodization.groups == [[1, 3, 5, 7], [2, 4, 6, 8]]
    @test count(periodization) == 4

    m = matrix(TBA(Lattice(unitcell, (2, 2), ('O', 'O')), hilbert, t))
    k = rand(2)
    U = matrix(periodization, k)
    @test periodization(m, k) ≈ U*m*U' ≈ [cos(k[1])+cos(k[2]) 0; 0 cos(k[1])+cos(k[2])]
end

@testset "EDSolver" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (2, 2), ('P', 'P'))
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site in eachindex(lattice))
    t, U = Hopping(:t, 1.0, 1), Hubbard(:U, 0.0)
    quantumnumber = ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)
    m = matrix(TBA(Lattice(unitcell, (2, 2), ('O', 'O')), hilbert, t))

    solver = ImpuritySolver(lattice, hilbert, (t, U), quantumnumber)
    @test valtype(solver) == Matrix{ComplexF64}

    ω = rand(ComplexF64)
    G, G⁻¹, ∂G = solver(ω), inv(solver, ω), derivative(solver, ω)
    @test solver(ω) === G ≈ inv(ω*I-m)
    @test inv(solver, ω) === G⁻¹ ≈ invoke(inv, Tuple{ImpuritySolver, Number}, solver, ω) ≈ ω*I-m
    @test derivative(solver, ω) === ∂G ≈ -inv(ω*I-m)^2

    ω′ = rand(ComplexF64)
    G′, G⁻¹′, ∂G′, workspace = solver.G.data, solver.G⁻¹.data, solver.∂G.data, solver.workspace
    @test solver(ω′) === G′ == G ≈ inv(ω′*I-m)
    @test inv(solver, ω′) === G⁻¹′ === G⁻¹ ≈ ω′*I-m
    @test derivative(solver, ω′) === ∂G′ === ∂G ≈ -inv(ω′*I-m)^2
    @test solver.workspace === workspace
    allocated_derivative(solver, ω) = @allocated derivative(solver, ω)
    @test allocated_derivative(solver, ω′+0.1im) == 0

    ω′′ = rand(ComplexF64)
    G⁻¹′′ = inv(solver, ω′′)
    @test !(solver.G.ω ≈ solver.G⁻¹.ω)
    @test G⁻¹′′ === G⁻¹′ === G⁻¹ ≈ ω′′*I-m
    @test solver.workspace === workspace

    borrowed = nothing
    with_solver(solver) do
        borrowed = current_solver(solver)
        @test borrowed !== solver
        @test borrowed.gf === solver.gf
        @test borrowed.pool === solver.pool
        @test solver(ω′) === borrowed.G.data
        @test inv(solver, ω′) === borrowed.G⁻¹.data
        @test derivative(solver, ω′) === borrowed.∂G.data
        @test allocated_derivative(solver, ω′+0.1im) == 0
    end
    with_solver(solver) do
        @test current_solver(solver) === borrowed
    end
    @test solver.pool.active == 0

    β = 2.0
    mktempdir() do dir
        cd(dir) do
            method = ThermalSeedBandLanczosMethod(β=β, cutoff=0.9, nrandom=3, maxdim₁=40, maxdim₂=64, seed=UInt64(1234))
            solver = ImpuritySolver(lattice, hilbert, (t, U), quantumnumber, method)
            solver(ω′)
            qlsave(pathof(solver)*".qlc", stamp(solver; ndecimal=14), (Ω=0.0, gf=solver.gf, samples=solver.method.samples))
            reloaded = ImpuritySolver(lattice, hilbert, (t, U), quantumnumber, method)
            reloaded(ω′)
            @test reloaded.Ω ≈ 0.0
            @test reloaded.gf == solver.gf
        end
    end
end

@testset "Non-interacting CPT: TBA case" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (2, 2), ('P', 'P'))
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site in eachindex(lattice))
    t = Hopping(:t, -1.0, 1)
    quantumnumber = ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)
    cpt = Algorithm(:FreeSquare, CPT(unitcell, lattice, hilbert, t, quantumnumber, BandLanczosMethod()))

    @testset "Bare TBA" begin
        tba = TBA(cpt)
        ref = TBA(unitcell, Hilbert(Fock{:f}(1, 2), length(unitcell)), t)
        @test kind(tba) == kind(ref)
        for k in ([0.13π, 0.47π], [0.83π, π], [0.3, 0.1])
            @test matrix(tba, k) == matrix(ref, k)
        end
    end

    @testset "Filling and chemical potential" begin
        filling_bz = BrillouinZone(unitcell, 20)
        @test filling(cpt; μ=0.3, brillouinzone=filling_bz)≈1.155 atol=1e-3
        @test chemicalpotential(cpt, 0.7; brillouinzone=filling_bz)≈-0.72654 atol=1e-2
        @test filling(cpt; μ=-0.72654, brillouinzone=filling_bz)≈0.7 atol=1e-2
    end

    @testset "Dynamical Spectra" begin
        η = 0.1
        ω = 0.7
        bz = BrillouinZone(unitcell, (4, 3))
        ss = cpt(:FreeDynamicalSurface, DynamicalSpectra(bz, ω, [1], [2], :; η=η))
        @test size(ss.data.values) == (1, 12, 3)
        @test ss.data.values[1, :, 1] ≈ ss.data.values[1, :, 2]
        @test ss.data.values[1, :, 1] + ss.data.values[1, :, 2] ≈ ss.data.values[1, :, 3]
        @test ss.data.values[1, :, 3] ≈ [2η/((ω+2cos(k[1])+2cos(k[2]))^2+η^2)/π for k in bz]

        ss′ = cpt(:FreeDynamicalSurface′, DynamicalSpectra(bz, ω; η=η); ntasks=2)
        @test size(ss′.data.values) == (1, 12, 1)
        @test ss′.data.values[1, :, 1] ≈ ss.data.values[1, :, 3]
        @test cpt.frontend.solver.pool.active == 0

        @test_throws ErrorException DynamicalSpectra(bz, ω; η=0.0)
        @test_throws ErrorException DynamicalSpectra(bz, ω; η=-0.1)
        @test_throws ErrorException DynamicalSpectra(bz, [ω, 2ω]; η=0.0)
        @test_throws ErrorException cpt(:FreeDynamicalSurface, DynamicalSpectra(bz, ω, [3]; η=η))
        @test_throws ErrorException cpt(:FreeDynamicalSurface, DynamicalSpectra(bz, ω, Int[]; η=η))
        @test_throws AssertionError cpt(:FreeDynamicalSurface, DynamicalSpectra(bz, ω; η=η); gauge=:icoordinate)
    end

    @testset "Density of States" begin
        η = 0.1
        es = LinRange(-4.0, 4.0, 401)
        bz = BrillouinZone(unitcell, (4, 3))
        dos = cpt(:FreeDOS, DensityOfStates(bz, es, [1], [2], :; η=η))
        @test size(dos.data.values) == (401, 3)
        @test dos.data.values[:, 1] ≈ dos.data.values[:, 2]
        @test dos.data.values[:, 3] ≈ dos.data.values[:, 1] + dos.data.values[:, 2]
        @test dos.data.values[:, 3] ≈ [mapreduce(k->2η/((ω+2cos(k[1])+2cos(k[2]))^2+η^2), +, bz)*step(es)/(π*length(bz)) for ω in es]

        dos′ = cpt(:FreeDOS′, DensityOfStates(bz, es; η=η); ntasks=2)
        @test size(dos′.data.values) == (401, 1)
        @test dos′.data.values[:, 1] ≈ dos.data.values[:, 3]
        @test cpt.frontend.solver.pool.active == 0

        # constructor-time and run-time validation
        @test_throws ErrorException DensityOfStates(bz, [0.0])
        @test_throws ErrorException DensityOfStates(bz, LinRange(1.0, -1.0, 10))
        @test_throws ErrorException DensityOfStates(bz, [0.0, 1.0, 1.5, 3.0])
        @test_throws ErrorException DensityOfStates(bz, es; η=0.0)
        @test_throws ErrorException DensityOfStates(bz, es; η=-0.1)
        @test_throws AssertionError cpt(:FreeDOSOpt, DensityOfStates(bz, es, :; η=η); gauge=:icoordinate)
        @test_throws AssertionError cpt(:FreeDOSTasks, DensityOfStates(bz, es; η=η); ntasks=0)
    end
end

@testset "Non-interacting CPT: BdG case" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (2, 2), ('P', 'P'))
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site in eachindex(lattice))
    t = Hopping(:t, -1.0, 1)
    Δ = Pairing(:Δ, 0.2, 1, 𝕔𝕔(:, :, real(1im*σʸ)); amplitude=bond::Bond -> real(exp(2im*azimuth(rcoordinate(bond)))))
    μ = Onsite(:μ, 0.0)
    quantumnumber = 𝕊ᶻ(0)
    cpt = Algorithm(:FreeBdG, CPT(unitcell, lattice, hilbert, (t, μ, Δ), quantumnumber, BandLanczosMethod()))

    @testset "Filling" begin
        bz = BrillouinZone(unitcell, 8)
        @test isapprox(filling(cpt; brillouinzone=bz, maxevals=100), 1.0; atol=5e-4)
        update!(cpt; μ=-0.5)
        @test isapprox(filling(cpt; brillouinzone=bz, maxevals=100), 1.2125519119423727; atol=1e-6)
    end

    tba = TBA(cpt)

    @testset "Dynamical Spectral" begin
        η = 0.05
        ω = 0.3
        bz = BrillouinZone(unitcell, (6, 4))
        ref = zeros(length(bz))
        for (j, k) in enumerate(bz)
            values, vectors = eigen(tba, k)
            for n in eachindex(values)
                w = abs2(vectors[1, n]) + abs2(vectors[2, n])
                ref[j] += w*η/((ω-values[n])^2+η^2)
            end
        end
        surface = cpt(:FreeDynamicalSurface, DynamicalSpectra(bz, ω, [1, 2], [3, 4], :; η=η))
        @test size(surface.data.values) == (1, 24, 3)
        @test surface.data.values[1, :, 1] + surface.data.values[1, :, 2] ≈ surface.data.values[1, :, 3]
        @test surface.data.values[1, :, 1] ≈ ref/π
    end

    @testset "Density of States" begin
        η = 0.025
        es = LinRange(-4.0, 4.0, 401)
        bz = BrillouinZone(unitcell, 20)
        ref = zeros(length(es))
        for k in bz
            values, vectors = eigen(tba, k)
            for n in eachindex(values)
                w = abs2(vectors[1, n]) + abs2(vectors[2, n])
                @. ref += w*η/((es-values[n])^2+η^2)
            end
        end
        ref .*= step(es)/(π*length(bz))
        dos = cpt(:FreeDOS, DensityOfStates(bz, es, [1, 2], [3, 4], :; η=η))
        @test size(dos.data.values) == (401, 3)
        @test dos.data.values[:, 2] ≈ reverse(dos.data.values[:, 1])
        @test dos.data.values[:, 1] + dos.data.values[:, 2] ≈ dos.data.values[:, 3]
        @test dos.data.values[:, 1] ≈ ref
    end
end

@testset "Square-Hubbard" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (2, 2), ('P', 'P'))
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site in eachindex(lattice))
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 8.0)
    μ = Onsite(:μ, -U.value/2)
    quantumnumber = ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)
    timer = TimerOutput()

    @testset "CPT" begin
        cpt = Algorithm(:SquareHubbard, CPT(unitcell, lattice, hilbert, (t, μ, U), quantumnumber; timer); timer)
        @test isapprox(Ω(cpt), -4.4444120; atol=1e-6)

        η = 0.1
        es = LinRange(-8.0, 8.0, 401)
        path = ReciprocalPath(unitcell, rectangle"Γ-X-M-Γ"; length=100)
        spectra = cpt(:EB, DynamicalSpectra(path, es; η=η))
        Plots.savefig(Plots.plot(spectra), "Plots-Hubbard-Square-2x2-EB.png")
        Makie.save("Makie-Hubbard-Square-2x2-EB.png", Makie.plot(spectra))

        bz = BrillouinZone(unitcell, 100)
        fs = cpt(:FS, DynamicalSpectra(bz, 0.0, [1], [2]; η=η))
        Plots.savefig(Plots.plot(fs), "Plots-Hubbard-Square-2x2-spectral-FS.png")
        Makie.save("Makie-Hubbard-Square-2x2-spectral-FS.png", Makie.plot(fs))

        dos = cpt(:DOS, DensityOfStates(bz, es; η=η))
        Plots.savefig(Plots.plot(dos), "Plots-Hubbard-Square-2x2-DOS.png")
        Makie.save("Makie-Hubbard-Square-2x2-DOS.png", Makie.plot(dos))
    end

    @testset "VCA: AFM" begin
        m = Onsite(:m, 0.2, 𝕔⁺𝕔(:, :, σᶻ); amplitude=bond::Bond -> real(exp(1im*dot((π, π), rcoordinate(bond)))))
        vca = Algorithm(:SquareHubbardAFM, VCA(unitcell, lattice, hilbert, (t, μ, U), m, quantumnumber; timer); timer)

        op = optimize!(vca; Ω_options=(brillouinzone=BrillouinZone(lattice, 50),))[2]
        @test isapprox(op.minimum, -4.4929112; atol=1e-6)
        @test isapprox(op.minimizer[1], 0.19543; atol=1e-4)

        op = optimize!(update!(vca; m=0.2); method=NoisyNewton(), Ω_options=(brillouinzone=BrillouinZone(lattice, 50),))[2]
        @test isapprox(op.minimum, -4.4929112; atol=1e-6)
        @test isapprox(op.minimizer[1], 0.19553; atol=1e-4)

        M = [
            -1.0  0.0  0.0   0.0  0.0   0.0   0.0  0.0;
                0.0  1.0  0.0   0.0  0.0   0.0   0.0  0.0;
                0.0  0.0  1.0   0.0  0.0   0.0   0.0  0.0;
                0.0  0.0  0.0  -1.0  0.0   0.0   0.0  0.0;
                0.0  0.0  0.0   0.0  1.0   0.0   0.0  0.0;
                0.0  0.0  0.0   0.0  0.0  -1.0   0.0  0.0;
                0.0  0.0  0.0   0.0  0.0   0.0  -1.0  0.0;
                0.0  0.0  0.0   0.0  0.0   0.0   0.0  1.0
        ]
        @test isapprox(expectation(vca, M), -0.80705; atol=1e-4)
        @test isapprox(expectation(vca, :m), -0.80705; atol=1e-4)
        @test isapprox(filling(vca), 1.0; atol=1e-3)
    end

    @testset "VCA: dSC" begin
        Δ = Pairing(:Δ, 0.2, 1, 𝕔𝕔(:, :, real(1im*σʸ)); amplitude=bond::Bond -> real(exp(2im*azimuth(rcoordinate(bond)))))
        μ = Onsite(:μ, -1.2)
        quantumnumber = 𝕊ᶻ(0)
        vca = Algorithm(:SquareHubbardDSC, VCA(unitcell, lattice, hilbert, (t, μ, U), Δ, quantumnumber; timer); timer)
        op = optimize!(vca; Ω_options=(brillouinzone=BrillouinZone(lattice, 50),))[2]
        @test isapprox(op.minimum, -1.8009763; atol=1e-6)
        @test isapprox(op.minimizer[1], 0.14987; atol=1e-4)
    end
end

@testset "Operator-based CPT/VCA" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (2, 2), ('P', 'P'))
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 8.0)
    μ = Onsite(:μ, -U.value/2)
    m = Onsite(:m, 0.2, 𝕔⁺𝕔(:, :, σᶻ); amplitude=bond::Bond -> real(exp(1im*dot((π, π), rcoordinate(bond)))))
    quantumnumber = ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)
    operators = expand(OperatorGenerator(bonds(unitcell, 1), Hilbert(Fock{:f}(1, 2), length(unitcell)), (t, μ)))

    # Operator-based CPT/VCA: all rank-2 terms (t, μ) as unitcell operators, only rank-4 (U) as interactions
    cpt = CPT(unitcell, operators, lattice, hilbert, U, quantumnumber)
    @test isapprox(Ω(cpt), -4.4444120; atol=1e-6)
    vca = VCA(unitcell, operators, lattice, hilbert, U, m, quantumnumber)
    @test isapprox(Ω(vca), -4.4929057; atol=1e-6)
end

@testset "Periodic-boundary cluster" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (2, 2), ('P', 'P'))
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site in eachindex(lattice))
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 8.0)
    μ = Onsite(:μ, -U.value/2)

    k = rand(2)
    pert_obc = Perturbation(lattice, hilbert, (t, μ, U))
    pert_pbc = Perturbation(lattice, hilbert, (t, μ, U); periodic=true)
    @test pert_pbc() ≈ zeros(size(pert_obc()))
    @test pert_pbc(k) ≈ pert_obc(k) - matrix(pert_obc.intercluster)

    ω = rand(ComplexF64)
    quantumnumber = ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)
    solver = ImpuritySolver(lattice, hilbert, (t, μ, U), quantumnumber; periodic=true)
    ref = ImpuritySolver(ED(lattice, hilbert, (t, μ, U), quantumnumber), operators(Fermionic(:TBA), lattice, hilbert))
    @test solver(ω) ≈ ref(ω)

    cpt = Algorithm(:SquarePBC, CPT(unitcell, lattice, hilbert, (t, μ, U), quantumnumber; periodic=true))
    ops = expand(OperatorGenerator(bonds(unitcell, 1), Hilbert(Fock{:f}(1, 2), length(unitcell)), (t, μ)))
    cpt′ = CPT(unitcell, ops, lattice, hilbert, U, quantumnumber; periodic=true)
    @test isapprox(Ω(cpt), -4.388205904059097; atol=1e-6)
    @test isapprox(Ω(cpt′), -4.388205904059097; atol=1e-6)

    update!(cpt; μ=0.0, U=0.0)
    ops = expand(OperatorGenerator(bonds(unitcell, 1), Hilbert(Fock{:f}(1, 2), length(unitcell)), (t, zero(μ))))
    cpt′ = CPT(unitcell, ops, lattice, hilbert, zero(U), quantumnumber; periodic=true)
    @test cpt(ω, k) ≈ cpt′(ω, k) ≈ inv(ω*I+[2cos(k[1])+2cos(k[2]) 0; 0 2cos(k[1])+2cos(k[2])])
    @test isapprox(Ω(cpt), -1.6210722; atol=1e-6)
    @test isapprox(Ω(cpt′), -1.6210722; atol=1e-6)
end

@testset "Haldane-Hubbard" begin
    parameters = (t=Complex(-1.0), t′=Complex(-0.2), U=4.0)
    @inline parammap(parameters::NamedTuple) = (μ=-parameters.U/2, U=parameters.U)

    unitcell = Lattice([0.0, 0.0], [0.0, √3/3]; vectors=[[1.0, 0.0], [0.5, √3/2]])
    lattice = Lattice(
        [0.0, 0.0], [0.0, √3/3], [0.5, √3/2], [0.5, -√3/6], [1.0, 0.0], [1.0, √3/3];
        vectors=[[1.5, √3/2], [1.5, -√3/2]]
    )
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))

    t = Hopping(:t, parameters.t, 1; ismodulatable=false)
    t′ = Hopping(:t′, parameters.t′, 2;
        amplitude=bond::Bond->1im*cos(3*azimuth(rcoordinate(bond)))*(-1)^(bond[1].site%2),
        ismodulatable=false
    )
    μ = Onsite(:μ, parammap(parameters).μ)
    U = Hubbard(:U, parammap(parameters).U)

    quantumnumber = ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)
    timer = TimerOutput()
    haldane = Algorithm(
        :HaldaneHubbard,
        CPT(unitcell, lattice, hilbert, (t, t′, μ, U), quantumnumber; timer),
        parameters,
        parammap;
        timer
    )

    es = LinRange(-4.0, 4.0, 201)
    path = ReciprocalPath(unitcell, hexagon"Γ-K-M-Γ"; length=100)
    update!(haldane; U=4.0)
    spectra = haldane(:EB, DynamicalSpectra(path, es; η=0.04))
    Plots.savefig(Plots.plot(spectra), "Plots-Haldane-Hubbard-Topological.png")
    Makie.save("Makie-Haldane-Hubbard-Topological.png", Makie.plot(spectra))

    update!(haldane; U=4.6)
    spectra = haldane(:EB, DynamicalSpectra(path, es; η=0.04))
    Plots.savefig(Plots.plot(spectra), "Plots-Haldane-Hubbard-Transition.png")
    Makie.save("Makie-Haldane-Hubbard-Transition.png", Makie.plot(spectra))

    update!(haldane; U=5.0)
    spectra = haldane(:EB, DynamicalSpectra(path, es; η=0.04))
    Plots.savefig(Plots.plot(spectra), "Plots-Haldane-Hubbard-Trivial.png")
    Makie.save("Makie-Haldane-Hubbard-Trivial.png", Makie.plot(spectra))

    num = 8
    edge = Lattice(lattice, (1, num), ('P', 'O'))
    hilbert_edge = Hilbert(Fock{:f}(1, 2), length(edge))
    haldane_edge = Algorithm(
        :HaldaneHubbardEdge,
        CPT(edge, edge, hilbert_edge, (t, t′, μ, U), ntuple(i->(6(i-1)+1, 6(i-1)+2, 6(i-1)+3, 6(i-1)+4, 6(i-1)+5, 6(i-1)+6), num)=>quantumnumber; timer),
        parameters,
        parammap;
        timer
    );
    update!(haldane_edge; U=4.0)
    @test Parameters(haldane_edge.frontend.solver) == (t=Complex(-1.0), t′=Complex(-0.2), μ=-2.0, U=4.0)
    ω = rand(ComplexF64)
    @test invoke(inv, Tuple{ImpuritySolver, Number}, haldane_edge.frontend.solver, ω) ≈ inv(haldane_edge.frontend.solver, ω)
    @test isapprox(Ω(haldane_edge.frontend.solver), -125.3539916; atol=1e-6)
    path_edge = ReciprocalPath(edge, -0.5=>0.5; length=100)
    spectra_edge = haldane_edge(:Edge, DynamicalSpectra(path_edge, es; η=0.04))
    Plots.savefig(Plots.plot(spectra_edge), "Plots-Haldane-Hubbard-Edge-Topological.png")
    Makie.save("Makie-Haldane-Hubbard-Edge-Topological.png", Makie.plot(spectra_edge))

    update!(haldane_edge; U=5.0)
    spectra_edge = haldane_edge(:Edge, DynamicalSpectra(path_edge, es; η=0.04))
    Plots.savefig(Plots.plot(spectra_edge), "Plots-Haldane-Hubbard-Edge-Trivial.png")
    Makie.save("Makie-Haldane-Hubbard-Edge-Trivial.png", Makie.plot(spectra_edge))
end

@testset "Partitioned VCA" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (2, 4), ('P', 'P'))
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site in eachindex(lattice))
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 8.0)
    μ = Onsite(:μ, -U.value/2)
    m = Onsite(:m, 0.19543, 𝕔⁺𝕔(:, :, σᶻ); amplitude=bond::Bond -> real(exp(1im*dot((π, π), rcoordinate(bond)))))
    timer = TimerOutput()

    vca = Algorithm(:Partitioned, VCA(unitcell, lattice, hilbert, (t, μ, U), m, ((1, 2, 3, 4), (5, 6, 7, 8)) => ℕ(4) ⊠ 𝕊ᶻ(0); timer); timer)

    solver = vca.frontend.solver
    ω, δ = 0.4 + 0.6im, 1e-6
    G, G⁻¹, ∂G = solver(ω), inv(solver, ω), derivative(solver, ω)
    @test solver(ω) === G
    @test inv(solver, ω) === G⁻¹ ≈ inv(G)
    @test derivative(solver, ω) === ∂G
    @test ∂G ≈ (copy(solver(ω+δ))-copy(solver(ω-δ)))/(2δ) atol=1e-5

    @test Ω(vca) ≈ -4.4929112 atol=1e-6

    task = DensityOfStates(BrillouinZone(unitcell, 2), LinRange(-1.0, 1.0, 5), [1, 2]; η=0.1)
    dos = vca(:DOS, task)
    dos′ = vca(:DOS′, task; ntasks=2)
    @test dos′.data.values ≈ dos.data.values
    @test vca.frontend.solver.pool.active == 0
end

@testset "Thermal CPT" begin
    μ = 0.3
    η = 0.05

    unitcell = Lattice([0.0]; vectors=[[1.0]])
    lattice = Lattice(unitcell, (10,), ('P',))
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
    terms = (Hopping(:t, -1.0, 1), Onsite(:mu, -μ))
    qn = ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)
    thermal = Algorithm(:Chain, CPT(unitcell, lattice, hilbert, terms, qn, ThermalSeedBandLanczosMethod(β=50.0, nrandom=1, maxdim₁=200, maxdim₂=200, seed=UInt64(1234))))

    ks = [0.13π, 0.37π, 0.5π, 0.71π, 0.91π, 1.0π]
    ωs = LinRange(-4.0, 4.0, 81)
    for k in ks, ω in ωs
        z = ω + 1im*η
        ε = -2cos(k) - μ
        ref = [1/(z-ε) 0; 0 1/(z-ε)]
        G = thermal(z, [k])
        @test maximum(abs.(G - ref)) < 1e-8
    end
end
