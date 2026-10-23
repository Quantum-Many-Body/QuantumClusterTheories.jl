using ExactDiagonalization
using LinearAlgebra: I, dot, inv, tr
using QuantumClusterTheories
using QuantumLattices
using TightBindingApproximation
using TimerOutputs: TimerOutput
import CairoMakie as Makie
import Plots

@testset "Basics" begin
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

    terms = (Hopping(:t, 1.0, 1), Hubbard(:U, 0.0))
    @test quadratic(terms) == (terms[1],)

    pert = Perturbation(lattice, hilbert, terms)
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

    tba = TBA(Lattice(unitcell, (2, 2), ('O', 'O')), hilbert, first(terms))
    m = matrix(tba)
    k = rand(2)
    U = matrix(periodization, k)
    @test periodization(m, k) ≈ U*m*U' ≈ [cos(k[1])+cos(k[2]) 0; 0 cos(k[1])+cos(k[2])]

    quantumnumber = ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)
    solver = ImpuritySolver(lattice, hilbert, terms, quantumnumber)
    @test valtype(solver) == Matrix{ComplexF64}
    ω = rand(ComplexF64)
    @test inv(ω*I-m) ≈ solver(ω)
    @test invoke(inv, Tuple{ImpuritySolver, Number}, solver, ω) ≈ inv(solver, ω)

    # the frequency derivative of the solver Green's function, cached per frequency
    ∂G = derivative(solver, ω)
    @test ∂G ≈ -inv(ω*I-m)^2
    @test derivative(solver, ω)===∂G
    ω′ = rand(ComplexF64)
    @test derivative(solver, ω′) ≈ -inv(ω′*I-m)^2

    cpt = Algorithm(:Square, CPT(unitcell, lattice, hilbert, terms, quantumnumber))
    @test cpt(ω, k) ≈ inv(ω*I-[2cos(k[1])+2cos(k[2]) 0; 0 2cos(k[1])+2cos(k[2])])
    @test isapprox(Ω(cpt), -1.6210722; atol=1e-6)
end

@testset "Square-Hubbard-Spectral" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (2, 2), ('P', 'P'))
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site in eachindex(lattice))
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 8.0)
    μ = Onsite(:μ, -U.value/2)
    quantumnumber = ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)
    timer = TimerOutput()
    cpt = Algorithm(:SquareHubbard, CPT(unitcell, lattice, hilbert, (t, μ, U), quantumnumber; timer); timer)
    @test isapprox(Ω(cpt), -4.4444120; atol=1e-6)

    es = LinRange(-10.0, 10.0, 501)
    path = ReciprocalPath(unitcell, rectangle"Γ-X-M-Γ"; length=100)
    spectra = cpt(:EB, DynamicalSpectra(path, es; η=0.1))
    Plots.savefig(Plots.plot(spectra), "Plots-Hubbard-Square-2x2-spectral.png")
    Makie.save("Makie-Hubbard-Square-2x2-spectral.png", Makie.plot(spectra))
end

@testset "PBC cluster" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (2, 2), ('P', 'P'))
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site in eachindex(lattice))
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 8.0)
    μ = Onsite(:μ, -U.value/2)
    quantumnumber = ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)
    ω = rand(ComplexF64)
    k = rand(2)

    # Test A: the periodic perturbation subtracts the intercluster terms as a k-independent real-space matrix
    pert_obc = Perturbation(lattice, hilbert, (t, μ, U))
    pert_pbc = Perturbation(lattice, hilbert, (t, μ, U); periodic=true)
    @test pert_pbc(k) ≈ pert_obc(k) - matrix(pert_obc.intercluster)
    @test pert_pbc() ≈ zeros(size(pert_obc()))

    # Test B: the periodic impurity solver matches a direct PBC exact diagonalization of the cluster
    solver = ImpuritySolver(lattice, hilbert, (t, μ, U), quantumnumber; periodic=true)
    ed = ED(lattice, hilbert, (t, μ, U), quantumnumber)
    ref = ImpuritySolver(ed, operators(Fermionic(:TBA), lattice, hilbert))
    @test solver(ω) ≈ ref(ω)

    # Test C: without interactions (U=0), periodic CPT recovers the exact bare Green's function and grand potential
    terms₀ = (Hopping(:t, 1.0, 1), Hubbard(:U, 0.0))
    cpt₀ = Algorithm(:SquarePBC, CPT(unitcell, lattice, hilbert, terms₀, quantumnumber; periodic=true))
    @test cpt₀(ω, k) ≈ inv(ω*I-[2cos(k[1])+2cos(k[2]) 0; 0 2cos(k[1])+2cos(k[2])])
    @test isapprox(Ω(cpt₀), -1.6210722; atol=1e-6)

    # Test D: with interactions, the term-based and operator-based constructions agree, and differ from open boundaries
    cpt = CPT(unitcell, lattice, hilbert, (t, μ, U), quantumnumber; periodic=true)
    ops = expand(OperatorGenerator(bonds(unitcell, 1), Hilbert(Fock{:f}(1, 2), length(unitcell)), (t, μ)))
    cpt′ = CPT(unitcell, ops, lattice, hilbert, U, quantumnumber; periodic=true)
    @test isapprox(Ω(cpt), Ω(cpt′); atol=1e-6)
    @test !isapprox(Ω(cpt), -4.4444120; atol=1e-6)
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

@testset "Square-Hubbard-AFM" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (2, 2), ('P', 'P'))
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site in eachindex(lattice))
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 8.0)
    m = Onsite(:m, 0.2, 𝕔⁺𝕔(:, :, σᶻ); amplitude=bond::Bond -> real(exp(1im*dot((π, π), rcoordinate(bond)))))
    μ = Onsite(:μ, -U.value/2)
    quantumnumber = ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)
    timer = TimerOutput()
    vca = Algorithm(:SquareHubbard, VCA(unitcell, lattice, hilbert, (t, μ, U), m, quantumnumber; timer); timer)

    op = optimize!(vca)[2]
    @test isapprox(op.minimum, -4.4929112; atol=1e-6)
    @test isapprox(op.minimizer[1], 0.19552; atol=1e-4)

    op = optimize!(update!(vca; m=0.2); method=NoisyNewton())[2]
    @test isapprox(op.minimum, -4.4929112; atol=1e-6)
    @test isapprox(op.minimizer[1], 0.19552; atol=1e-4)

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

    # vs = LinRange(0.0, 0.3, 31)
    # result = zeros(length(vs))
    # for (i, v) in enumerate(vs)
    #     update!(vca; m=v)
    #     result[i] = Ω(vca)
    # end
    # Plots.savefig(Plots.plot(vs, result), "Plots-Square-Hubbard-AFM.png")
    # Makie.save("Makie-Square-Hubbard-AFM.png", Makie.lines(vs, result))
end

@testset "Square-Hubbard-dSC" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (2, 2), ('P', 'P'))
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site in eachindex(lattice))
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 8.0)
    Δ = Pairing(:Δ, 0.2, 1, 𝕔𝕔(:, :, real(1im*σʸ)); amplitude=bond::Bond -> real(exp(2im*azimuth(rcoordinate(bond)))))
    μ = Onsite(:μ, -1.2)
    quantumnumber = 𝕊ᶻ(0)
    timer = TimerOutput()
    vca = Algorithm(:SquareHubbard, VCA(unitcell, lattice, hilbert, (t, μ, U), Δ, quantumnumber; timer); timer)
    op = optimize!(vca)[2]
    @test isapprox(op.minimum, -1.8009791; atol=1e-6)
    @test isapprox(op.minimizer[1], 0.14984; atol=1e-4)

    # vs = LinRange(0.0, 0.3, 31)
    # result = zeros(length(vs))
    # for (i, v) in enumerate(vs)
    #     update!(vca; Δ=v)
    #     result[i] = Ω(vca)
    # end
    # Plots.savefig(Plots.plot(vs, result), "Plots-Square-Hubbard-dSC.png")
    # Makie.save("Makie-Square-Hubbard-dSC.png", Makie.lines(vs, result))
end

@testset "Partitioned VCA" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (2, 4), ('P', 'P'))
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site in eachindex(lattice))
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 8.0)
    μ = Onsite(:μ, -U.value/2)
    m = Onsite(:m, 0.2, 𝕔⁺𝕔(:, :, σᶻ); amplitude=bond::Bond -> real(exp(1im*dot((π, π), rcoordinate(bond)))))
    partition = ((1, 2, 3, 4), (5, 6, 7, 8)) => ℕ(4) ⊠ 𝕊ᶻ(0)
    timer = TimerOutput()

    # Vanishing Weiss field: partitioned VCA reduces to partitioned CPT
    cpt = CPT(unitcell, lattice, hilbert, (t, μ, U), partition; timer)
    vca₀ = VCA(unitcell, lattice, hilbert, (t, μ, U), zero(m), partition; timer)
    ω, k = rand(ComplexF64), rand(2)
    @test vca₀(ω, k) ≈ cpt(ω, k)
    @test isapprox(Ω(vca₀), Ω(cpt); atol=1e-6)
    @test isapprox(Ω(cpt), -4.4444120; atol=1e-6)

    # the frequency derivative of the composed solver Green's function, cached per frequency (fixed ω with O(1) imaginary part keeps the finite difference well conditioned)
    solver = cpt.solver
    ω = 0.3 + 0.5im
    ∂G = derivative(solver, ω)
    @test derivative(solver, ω)===∂G
    δ = 1e-6
    @test isapprox(∂G, (copy(solver(ω+δ))-copy(solver(ω-δ)))/(2δ); atol=1e-5)

    # Variational optimization of the staggered Weiss field
    vca = Algorithm(:SquareHubbardPlaquette, VCA(unitcell, lattice, hilbert, (t, μ, U), m, partition; timer); timer)
    op = optimize!(vca)[2]
    @test isapprox(op.minimum, -4.4929112; atol=1e-6)
    @test isapprox(op.minimizer[1], 0.19543; atol=1e-4)
    @test op.minimum < Ω(cpt)
end

@testset "ThermalCPT" begin
    μ = 0.3
    η = 0.05

    unitcell = Lattice([0.0]; vectors=[[1.0]])
    lattice = Lattice(unitcell, (10,), ('P',))
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
    terms = (Hopping(:t, -1.0, 1), Onsite(:mu, -μ))
    qn = ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)
    thermal = Algorithm(:Chain10, CPT(unitcell, lattice, hilbert, terms, qn, ThermalSeedBandLanczosMethod(β=50.0, nrandom=1, maxdim₁=200, maxdim₂=200, seed=UInt64(1234))))
    zero = Algorithm(:Chain10, CPT(unitcell, lattice, hilbert, terms, qn, BandLanczosMethod()))

    ks = [0.13π, 0.37π, 0.5π, 0.71π, 0.91π, 1.0π]
    ωs = LinRange(-4.0, 4.0, 81)
    for k in ks, ω in ωs
        z = ω + 1im*η
        ε = -2cos(k) - μ
        ref = [1/(z-ε) 0; 0 1/(z-ε)]
        G = thermal(z, [k])
        @test maximum(abs.(G - ref)) < 1e-8
        @test maximum(abs.(G - zero(z, [k]))) < 1e-8
    end

    # spectral sum rule per momentum: ∫dω (-1/π)Im tr G(k, ω) = 2 (both spins)
    ωs = LinRange(-6.0, 6.0, 1201)
    W = 6.0
    for k in ks
        ε = -2cos(k) - μ
        vals = [-imag(tr(thermal(ω + 1im*η, [k])))/π for ω in ωs]
        A = (sum(vals)-(first(vals)+last(vals))/2) * step(ωs)
        ref = 2*(atan((W-ε)/η) + atan((W+ε)/η))/π
        @test isapprox(A, ref; atol=1e-6)
    end
end

@testset "TBA" begin
    # TBA(::QCT): the bare unitcell tight-binding system reconstructed from a QCT frontend coincides with the directly constructed one
    unitcell = Lattice([0.0]; vectors=[[1.0]])
    lattice = Lattice(unitcell, (4,), ('P',))
    terms = (Hopping(:t, -1.0, 1), Onsite(:mu, -0.3))
    cpt = CPT(unitcell, lattice, Hilbert(Fock{:f}(1, 2), length(lattice)), terms, ℕ(length(lattice)) ⊠ 𝕊ᶻ(0), BandLanczosMethod())
    tba, ref = TBA(cpt), TBA(unitcell, Hilbert(Fock{:f}(1, 2), length(unitcell)), terms)
    @test kind(tba)==kind(ref)
    for k in ([0.13π], [0.47π], [0.83π], [π])
        @test matrix(tba, k)==matrix(ref, k)
        @test velocity(tba, k, 1)==velocity(ref, k, 1)
    end

    unitcell = Lattice([0.0, 0.0], [0.0, √3/3]; vectors=[[1.0, 0.0], [0.5, √3/2]])
    lattice = Lattice(unitcell, (2, 2), ('P', 'P'))
    terms = (
        Hopping(:t, Complex(-1.0), 1),
        Hopping(:t′, Complex(-0.2), 2; amplitude=bond::Bond->1im*cos(3*azimuth(rcoordinate(bond)))*(-1)^(bond[1].site%2))
    )
    cpt = Algorithm(:Haldane, CPT(unitcell, lattice, Hilbert(Fock{:f}(1, 2), length(lattice)), terms, ℕ(length(lattice)) ⊠ 𝕊ᶻ(0), BandLanczosMethod()))
    tba, ref = TBA(cpt), TBA(unitcell, Hilbert(Fock{:f}(1, 2), length(unitcell)), terms)
    @test kind(tba)==kind(ref)
    for k in ([0.3, 0.1], [1.1, -0.7], [2.5, 2.0])
        @test matrix(tba, k)==matrix(ref, k)
        @test velocity(tba, k, 1)==velocity(ref, k, 1)
        @test velocity(tba, k, 2)==velocity(ref, k, 2)
    end
end
