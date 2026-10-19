using ExactDiagonalization
using LinearAlgebra: I, inv, tr
using QuantumClusterTheories
using QuantumLattices
using TightBindingApproximation
using QuadGK

# Benchmarks of the dc conductivity module (src/Transport.jl), all in units of e²/h per layer:
# B1: longitudinal σ_xx of the free 1D chain vs the analytic Lorentzian-Boltzmann result.
# B2: transverse σ_xy of the Haldane Chern insulator vs the quantized value σ_xy = +C = 2.
# B3: self-consistency on the Haldane metal — at U=0 CPT is exact for any cluster, so the 6-site-cluster conductivity must coincide with the unitcell-cluster one.

@testset "Transport-B1-1DChain" begin
    # longitudinal conductivity of the 1D free spin-1/2 fermion chain at β=50 (thermal seed, nr=1), with the chemical potential supplied by the Conductivity task (the Hamiltonian is μ-free):
    #
    #   σ_xx = (2π²e²/Nh)Σ_k ∫dω (-∂_ωf) Tr[v_x A v_x A] = (4π²e²/Nh)Σ_k (2t sin k)² ∫dω (-∂_ωf) A_L(ω-ε_k)²
    #
    # with the Lorentzian A_L(x) = (η/π)/(x²+η²), ε_k = -2t·cos k - μ and v_x = ∂_kε_k.
    unitcell = Lattice([0.0]; vectors=[[1.0]])
    lattice = Lattice(unitcell, (10,), ('P',))
    hilbert = Hilbert(Fock{:f}(1, 2), length(lattice))
    μ = 0.3
    terms = (Hopping(:t, -1.0, 1),)
    qn = ℕ(length(lattice)) ⊠ 𝕊ᶻ(0)

    β, η = 50.0, 0.05
    thermal = Algorithm(:Chain, CPT(unitcell, lattice, hilbert, terms, qn, ThermalSeedBandLanczosMethod(β=β, nrandom=1, maxdim₁=200, maxdim₂=200, seed=UInt64(1234))))
    bz = BrillouinZone(unitcell, 512)
    data = thermal(:σ, Conductivity(bz; μ=μ, β=β, η=η, which=:surf)).data
    @test data.sea == zeros(1, 1)
    # analytic reference by independent high-precision quadrature of the known integrand
    ∂fermi(ω) = (x = β*ω/2; abs(x)>40 ? 0.0 : β/4*sech(x)^2)
    ks = [k[1] for k in bz]
    integrand(ω) = ∂fermi(ω)*sum(k -> (2sin(k)*((η/π)/((ω+2cos(k)+μ)^2+η^2)))^2, ks)
    ref = (4π^2/length(ks))*quadgk(integrand, -Inf, Inf; rtol=1e-10)[1]
    @test isapprox(data.surf[1, 1], ref; rtol=2e-4, atol=1e-4)
    @test data.surf[1, 1] == data.total[1, 1]
end

# the Haldane model building blocks (U=0)
function haldane_terms(t::Real, t′::Real, μ::Real)
    return (
        Hopping(:t, Complex(-t), 1),
        Hopping(:t′, Complex(-t′), 2; amplitude=bond::Bond->1im*cos(3*azimuth(rcoordinate(bond)))*(-1)^(bond[1].site%2)),
        Onsite(:μ, Complex(-μ))
    )
end
const haldane_unitcell = Lattice([0.0, 0.0], [0.0, √3/3]; vectors=[[1.0, 0.0], [0.5, √3/2]])
const haldane_lattice = Lattice(
    [0.0, 0.0], [0.0, √3/3], [0.5, √3/2], [0.5, -√3/6], [1.0, 0.0], [1.0, √3/3];
    vectors=[[1.5, √3/2], [1.5, -√3/2]]
)

@testset "Transport-B2-Haldane-Chern" begin
    # the Haldane Chern insulator (C=1 per spin, C=2 total): σ_xy quantized to C = 2 in units of e²/h
    hilbert = Hilbert(Fock{:f}(1, 2), length(haldane_lattice))
    qn = ℕ(length(haldane_lattice)) ⊠ 𝕊ᶻ(0)
    terms = haldane_terms(1.0, 0.2, 0.0)

    # zero temperature: by the band Lanczos method, computing both surf and sea terms
    cpt = Algorithm(:HaldaneChern, CPT(haldane_unitcell, haldane_lattice, hilbert, terms, qn, BandLanczosMethod()))
    bz = BrillouinZone(haldane_unitcell, 40)
    data = cpt(:σ, Conductivity(bz; β=Inf, η=0.02)).data
    # the total Chern number of the two occupied (spin-degenerate) bands is +2, so σ_xy = C = 2 in units of e²/h
    @test isapprox(data.total[1, 2], 2; atol=1e-3)
    @test isapprox(data.total[1, 2], -data.total[2, 1]; atol=1e-10)
    # no Fermi surface in the gap: the surf term vanishes up to the η-broadening tails (~1e-3 here)
    @test abs(data.surf[1, 1]) < 1e-2
    @test abs(data.surf[1, 2]) < 1e-6

    # finite temperature: the same Chern insulator probed by the thermal seed solver at β=30, computing the sea term alone
    cpt = Algorithm(:HaldaneChern, CPT(haldane_unitcell, haldane_lattice, hilbert, terms, qn, ThermalSeedBandLanczosMethod(β=30.0, nrandom=1, maxdim₁=200, maxdim₂=200, seed=UInt64(1234))))
    data = cpt(:σ, Conductivity(bz; β=30.0, η=0.02, which=:sea)).data
    @test isapprox(data.total[1, 2], 2; atol=1e-3)
    @test isapprox(data.total[1, 2], -data.total[2, 1]; atol=1e-10)
    @test data.surf == zeros(2, 2)
end

@testset "Transport-B3-Haldane-Metal" begin
    # self-consistency: at U=0 CPT is exact for any cluster, so the 6-site-cluster conductivity must coincide with the unitcell-cluster one, at zero and at finite temperature.
    hilbert = Hilbert(Fock{:f}(1, 2), length(haldane_lattice))
    hilbert_uc = Hilbert(Fock{:f}(1, 2), length(haldane_unitcell))
    qn = ℕ(length(haldane_lattice)) ⊠ 𝕊ᶻ(0)
    qn_uc = ℕ(length(haldane_unitcell)) ⊠ 𝕊ᶻ(0)
    terms = haldane_terms(1.0, 0.2, 0.0)
    μ, η = 1.0, 0.05
    bz = BrillouinZone(haldane_unitcell, 40)

    # zero temperature, metallic chemical potential μ = 1.0 (both σ^surf_xy and σ^sea_xy nonzero)
    cpt = Algorithm(:HaldaneMetal, CPT(haldane_unitcell, haldane_lattice, hilbert, terms, qn, BandLanczosMethod()))
    ref = Algorithm(:HaldaneMetalUC, CPT(haldane_unitcell, haldane_unitcell, hilbert_uc, terms, qn_uc, BandLanczosMethod()))
    data = cpt(:σ, Conductivity(bz; μ=μ, β=Inf, η=η); rtol=1e-8).data
    dataR = ref(:σ, Conductivity(bz; μ=μ, β=Inf, η=η); rtol=1e-8).data
    @test maximum(abs.(data.surf - dataR.surf)) < 1e-6
    @test maximum(abs.(data.sea - dataR.sea)) < 1e-5
    @test data.surf[1, 1] > 0.1  # metallic Drude weight
    @test isapprox(data.sea[1, 2], -data.sea[2, 1]; atol=1e-10)

    # finite temperature, metallic: the 2-site reference by the exact thermal method, the 6-site cluster by the thermal seed method (nr=1 at β=30)
    β = 30.0
    cpt = Algorithm(:HaldaneMetalT, CPT(haldane_unitcell, haldane_lattice, hilbert, terms, qn, ThermalSeedBandLanczosMethod(β=β, nrandom=1, maxdim₁=200, maxdim₂=200, seed=UInt64(1234))))
    ref = Algorithm(:HaldaneMetalTUC, CPT(haldane_unitcell, haldane_unitcell, hilbert_uc, terms, qn_uc, ThermalExactMethod(β=β)))
    data = cpt(:σ, Conductivity(bz; μ=μ, β=β, η=η); rtol=1e-8).data
    dataR = ref(:σ, Conductivity(bz; μ=μ, β=β, η=η); rtol=1e-8).data
    @test maximum(abs.(data.surf - dataR.surf)) < 1e-6
    @test maximum(abs.(data.sea - dataR.sea)) < 1e-5
end
