using ExactDiagonalization
using LinearAlgebra: I, inv, tr
using QuantumClusterTheories
using QuantumLattices
using Test
using TightBindingApproximation
import Plots

@testset "QuantumClusterTheories" begin
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

    pert = perturbation(lattice, hilbert, terms)
    @test matrix(pert) == [
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

    tba = TBA(Lattice(unitcell, (2, 2), ('O', 'O')), hilbert, first(terms))
    m = matrix(tba)
    k = rand(2)
    @test periodization(m, k) ≈ [cos(k[1])+cos(k[2]) 0; 0 cos(k[1])+cos(k[2])]

    quantumnumber = ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)
    solver = ImpuritySolver(lattice, hilbert, terms, quantumnumber)
    ω = rand(ComplexF64)
    @test inv(ω*I-m) ≈ solver(ω)

    cpt = CPT(unitcell, lattice, hilbert, terms, quantumnumber)
    @test cpt(ω, k) ≈ inv(ω*I-[2cos(k[1])+2cos(k[2]) 0; 0 2cos(k[1])+2cos(k[2])])
end

@testset begin "Square-Hubbard-Spectral"
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (2, 2), ('P', 'P'))
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site in eachindex(lattice))
    t = Hopping(:t, -1.0, 1)
    U = Hubbard(:U, 8.0)
    μ = Onsite(:μ, -U.value/2)
    quantumnumber = ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)
    cpt = Algorithm(:SquareHubbard, CPT(unitcell, lattice, hilbert, (t, μ, U), quantumnumber))

    es = LinRange(-10.0, 10.0, 501)
    path = ReciprocalPath(reciprocals(unitcell), rectangle"Γ-X-M-Γ"; length=100)
    spectra = cpt(:EB, DynamicalSpectra(path, es); η=0.1)
    Plots.savefig(Plots.plot(spectra), "Plots-Hubbard-Square-2x2-spectral.png")
end
