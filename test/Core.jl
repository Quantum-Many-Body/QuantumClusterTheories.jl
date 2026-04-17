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
    @test periodization(m, k) ≈ [cos(k[1])+cos(k[2]) 0; 0 cos(k[1])+cos(k[2])]

    quantumnumber = ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)
    solver = ImpuritySolver(lattice, hilbert, terms, quantumnumber)
    ω = rand(ComplexF64)
    @test inv(ω*I-m) ≈ solver(ω)
    @test invoke(inv, Tuple{ImpuritySolver, Number}, solver, ω) ≈ inv(solver, ω)

    cpt = CPT(unitcell, lattice, hilbert, terms, quantumnumber)
    @test cpt(ω, k) ≈ inv(ω*I-[2cos(k[1])+2cos(k[2]) 0; 0 2cos(k[1])+2cos(k[2])])
    @test isapprox(Ω(cpt), -1.621072213728453; atol=1e-6)
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
    @test isapprox(Ω(cpt), -4.4444120382788945; atol=1e-6)

    es = LinRange(-10.0, 10.0, 501)
    path = ReciprocalPath(reciprocals(unitcell), rectangle"Γ-X-M-Γ"; length=100)
    spectra = cpt(:EB, DynamicalSpectra(path, es); η=0.1)
    Plots.savefig(Plots.plot(spectra), "Plots-Hubbard-Square-2x2-spectral.png")
    Makie.save("Makie-Hubbard-Square-2x2-spectral.png", Makie.plot(spectra))
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
        CPT(unitcell, lattice, hilbert, (t, t′, μ, U), quantumnumber, BandLanczosMethod(keepvecs=true); timer),
        parameters,
        parammap;
        timer
    )

    es = LinRange(-4.0, 4.0, 201)
    path = ReciprocalPath(reciprocals(unitcell), hexagon"Γ-K-M-Γ"; length=100)
    update!(haldane; U=4.0)
    spectra = haldane(:EB, DynamicalSpectra(path, es); η=0.04)
    Plots.savefig(Plots.plot(spectra), "Plots-Haldane-Hubbard-Topological.png")
    Makie.save("Makie-Haldane-Hubbard-Topological.png", Makie.plot(spectra))

    update!(haldane; U=4.6)
    spectra = haldane(:EB, DynamicalSpectra(path, es); η=0.04)
    Plots.savefig(Plots.plot(spectra), "Plots-Haldane-Hubbard-Transition.png")
    Makie.save("Makie-Haldane-Hubbard-Transition.png", Makie.plot(spectra))

    update!(haldane; U=5.0)
    spectra = haldane(:EB, DynamicalSpectra(path, es); η=0.04)
    Plots.savefig(Plots.plot(spectra), "Plots-Haldane-Hubbard-Trivial.png")
    Makie.save("Makie-Haldane-Hubbard-Trivial.png", Makie.plot(spectra))

    num = 8
    edge = Lattice(lattice, (1, num), ('P', 'O'))
    hilbert_edge = Hilbert(Fock{:f}(1, 2), length(edge))
    haldane_edge = Algorithm(
        :HaldaneHubbardEdge,
        CPT(edge, edge, hilbert_edge, (t, t′, μ, U), ntuple(i->(6(i-1)+1, 6(i-1)+2, 6(i-1)+3, 6(i-1)+4, 6(i-1)+5, 6(i-1)+6), num)=>quantumnumber, BandLanczosMethod(keepvecs=true); timer),
        parameters,
        parammap;
        timer
    );
    update!(haldane_edge; U=4.0)
    @test Parameters(haldane_edge.frontend.solver) == (t=Complex(-1.0), t′=Complex(-0.2), μ=-2.0, U=4.0)
    ω = rand(ComplexF64)
    @test invoke(inv, Tuple{ImpuritySolver, Number}, haldane_edge.frontend.solver, ω) ≈ inv(haldane_edge.frontend.solver, ω)
    @test isapprox(Ω(haldane_edge.frontend.solver), -125.35399159356793; atol=1e-6)
    path_edge = ReciprocalPath(reciprocals(edge), -0.5=>0.5; length=100)
    spectra_edge = haldane_edge(:Edge, DynamicalSpectra(path_edge, es); η=0.04)
    Plots.savefig(Plots.plot(spectra_edge), "Plots-Haldane-Hubbard-Edge-Topological.png")
    Makie.save("Makie-Haldane-Hubbard-Edge-Topological.png", Makie.plot(spectra_edge))

    update!(haldane_edge; U=5.0)
    spectra_edge = haldane_edge(:Edge, DynamicalSpectra(path_edge, es); η=0.04)
    Plots.savefig(Plots.plot(spectra_edge), "Plots-Haldane-Hubbard-Edge-Trivial.png")
    Makie.save("Makie-Haldane-Hubbard-Edge-Trivial.png", Makie.plot(spectra_edge))
end

@testset "Square-Hubbard-AFM" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (2, 2), ('P', 'P'))
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site in eachindex(lattice))
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 8.0)
    m = Onsite(:m, 0.3, 𝕔⁺𝕔(:, :, σᶻ); amplitude=bond::Bond -> real(exp(1im*dot((π, π), rcoordinate(bond)))))
    μ = Onsite(:μ, -U.value/2)
    quantumnumber = ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)
    timer = TimerOutput()
    vca = Algorithm(:SquareHubbard, VCA(unitcell, lattice, hilbert, (t, μ, U), m, quantumnumber, BandLanczosMethod(keepvecs=true); timer); timer)

    op = optimize!(vca)[2]
    @test isapprox(op.minimum, -4.492911205682658; atol=1e-6)
    @test isapprox(op.minimizer[1], 0.1955178114114018; atol=1e-4)

    op = optimize!(update!(vca; m=0.2); method=NoisyNewton())[2]
    @test isapprox(op.minimum, -4.492911205682658; atol=1e-6)
    @test isapprox(op.minimizer[1], 0.1955178114114018; atol=1e-4)

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
    @test isapprox(expectation(vca, M), -0.8070466592810626; atol=1e-4)
    @test isapprox(expectation(vca, :m), -0.8070466592810626; atol=1e-4)

    vs = LinRange(0.0, 0.3, 31)
    result = zeros(length(vs))
    for (i, v) in enumerate(vs)
        update!(vca; m=v)
        result[i] = Ω(vca)
    end
    Plots.savefig(Plots.plot(vs, result), "Plots-Square-Hubbard-AFM.png")
    Makie.save("Makie-Square-Hubbard-AFM.png", Makie.lines(vs, result))
end
