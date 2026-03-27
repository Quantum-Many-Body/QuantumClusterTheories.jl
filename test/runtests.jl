using ExactDiagonalization
using LinearAlgebra: I, inv
using QuantumClusterTheories
using QuantumLattices
using Test
using TightBindingApproximation

@testset "QuantumClusterTheories" begin
    unitcell = Lattice([0.0, 0.0]; vectors=[[1.0, 0.0], [0.0, 1.0]])
    lattice = Lattice(unitcell, (2, 2), ('P', 'P'))
    hilbert = Hilbert(site=>Fock{:f}(1, 2) for site in eachindex(lattice))
    @test operators(Fermionic(:TBA), lattice, hilbert, Table(hilbert, Metric(Fermionic(:TBA), hilbert))) == [
        𝕔(1, 1, -1//2, [0.0, 0.0], [0.0, 0.0]), 𝕔(1, 1, 1//2, [0.0, 0.0], [0.0, 0.0]), 𝕔(2, 1, -1//2, [1.0, 0.0], [0.0, 0.0]), 𝕔(2, 1, 1//2, [1.0, 0.0], [0.0, 0.0]),
        𝕔(3, 1, -1//2, [0.0, 1.0], [0.0, 0.0]), 𝕔(3, 1, 1//2, [0.0, 1.0], [0.0, 0.0]), 𝕔(4, 1, -1//2, [1.0, 1.0], [0.0, 0.0]), 𝕔(4, 1, 1//2, [1.0, 1.0], [0.0, 0.0])
    ]
    @test operators(Fermionic(:BdG), lattice, hilbert, Table(hilbert, Metric(Fermionic(:BdG), hilbert))) == [
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

    ops_lattice = operators(Fermionic(:TBA), lattice, hilbert, Table(hilbert, Metric(Fermionic(:TBA), hilbert)))
    ops_unitcell = operators(Fermionic(:TBA), unitcell, hilbert, Table(hilbert, Metric(Fermionic(:TBA), hilbert)))
    periodization = Periodization(ops_lattice, ops_unitcell, unitcell.vectors)
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
