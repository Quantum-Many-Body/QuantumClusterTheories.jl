using LinearAlgebra: Diagonal, I, axpby!, dot, inv!, logabsdet, lu!, mul!, tr
using Roots: Brent, find_zero
using TimerOutputs: TimerOutput
using Optim: LBFGS, Options, optimize
using QuadGK: quadgk
using QuantumLattices: AbstractLattice, Algorithm, Assignment, Bond, BrillouinZone, CoordinatedIndex, Data, Fock, Generator, Hilbert, Index, LatticeModel, Metric, Neighbors, OneOrMore, ReciprocalSpace, Table, Term, ZeroAtLeast
using QuantumLattices: atol, bonds, expand, isannihilation, isintracell, issubordinate, nneighbor, rank, rcoordinate, rtol
using StaticArrays: SVector
using TightBindingApproximation: TBAKind

import QuantumLattices: Parameters, dependencytypes, kind, matrix, options, run!, update!
import TightBindingApproximation: TBA
import TightBindingApproximation.Fitting: optimize!

"""
    const qcttimer

Default shared timer for all quantum cluster theory methods.
"""
const qcttimer = TimerOutput()

"""
    operators(tbakind::TBAKind{:TBA}, lattice::AbstractLattice, hilbert::Hilbert{<:Fock}, table::Table=Table(hilbert, Metric(tbakind, hilbert))) -> Vector{<:CoordinatedIndex}
    operators(tbakind::TBAKind{:BdG}, lattice::AbstractLattice, hilbert::Hilbert{<:Fock}, table::Table=Table(hilbert, Metric(tbakind, hilbert))) -> Vector{<:CoordinatedIndex}

Get the single-particle operators sorted by table index.
For TBA kind, returns only annihilation operators; for BdG kind, returns all operators.
"""
function operators(tbakind::TBAKind{:TBA}, lattice::AbstractLattice, hilbert::Hilbert{<:Fock}, table::Table=Table(hilbert, Metric(tbakind, hilbert)))
    result = [CoordinatedIndex(Index(site, fockindex), coordinate, zero(coordinate)) for (site, coordinate) in enumerate(lattice) for fockindex in hilbert[site] if isannihilation(fockindex)]
    return sort!(result; by=index->table[index])
end
function operators(tbakind::TBAKind{:BdG}, lattice::AbstractLattice, hilbert::Hilbert{<:Fock}, table::Table=Table(hilbert, Metric(tbakind, hilbert)))
    result = [CoordinatedIndex(Index(site, fockindex), coordinate, zero(coordinate)) for (site, coordinate) in enumerate(lattice) for fockindex in hilbert[site]]
    return sort!(result; by=index->table[index])
end

"""
    quadratic(terms::ZeroAtLeast{Term}) -> Tuple

Extract the quadratic (rank-2) terms from a collection of terms.
"""
@generated quadratic(terms::ZeroAtLeast{Term}) = Expr(:tuple, [:(terms[$i]) for (i, T) in enumerate(fieldtypes(terms)) if rank(T)==2]...)

"""
    ImpuritySolver <: LatticeModel

Abstract type for impurity solvers used in quantum cluster theory calculations.
Subtypes must implement the call syntax `solver(ω)` to return the solver's response function at frequency `ω`.
"""
abstract type ImpuritySolver <: LatticeModel end
@inline Base.valtype(::Type{<:ImpuritySolver}) = Matrix{ComplexF64}

"""
    inv(solver::ImpuritySolver, ω::Number) -> Matrix{ComplexF64}

Get the inverse of the retarded Green's function at frequency `ω`
"""
@inline Base.inv(solver::ImpuritySolver, ω::Number) = inv(solver(ω))

"""
    derivative(solver::ImpuritySolver, ω::Number) -> Matrix{ComplexF64}

The derivative of the impurity solver's Green's function with respect to its frequency argument.
"""
@inline derivative(solver::ImpuritySolver, ::Number) = error("derivative is not implemented for $(typeof(solver)).")

"""
    Perturbation

Abstract type for perturbations in quantum cluster theory.
Subtypes must implement `kind`, `update!`, and the call syntax `perturbation(k)` returning the perturbation matrix at momentum `k`.
"""
abstract type Perturbation end
@inline kind(perturbation::Perturbation) = kind(typeof(perturbation))

"""
    CPTPerturbation{V<:TBA} <: Perturbation

CPT (cluster perturbation theory) perturbation containing the intercluster quadratic terms of a system.

When `periodic` is `true`, the intercluster terms are also subtracted as a momentum-independent real-space matrix, so that the same terms can be included in the impurity solver as periodic boundary conditions of the cluster without changing the total Hamiltonian.
"""
struct CPTPerturbation{V<:TBA} <: Perturbation
    intercluster::V
    periodic::Bool
end
@inline CPTPerturbation(intercluster::TBA) = CPTPerturbation(intercluster, false)
@inline kind(::Type{<:CPTPerturbation{V}}) where {V<:TBA} = kind(V)
@inline update!(perturbation::CPTPerturbation; parameters...) = (update!(perturbation.intercluster; parameters...); perturbation)
@inline function (perturbation::CPTPerturbation)(k::Union{AbstractVector{<:Number}, Nothing}=nothing)
    return perturbation.periodic ? matrix(perturbation.intercluster, k)-matrix(perturbation.intercluster) : matrix(perturbation.intercluster, k)
end

"""
    Perturbation(intercluster::TBA; periodic::Bool=false) -> CPTPerturbation
    Perturbation(lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}; neighbors::Union{Int, Neighbors}=nneighbor(terms), periodic::Bool=false) -> CPTPerturbation
    Perturbation(bonds::AbstractVector{<:Bond}, hilbert::Hilbert, terms::OneOrMore{Term}; periodic::Bool=false) -> CPTPerturbation

Construct a CPT perturbation from a pre-built `TBA`, or from a lattice/hilbert/terms, or from bonds/hilbert/terms.

`periodic=true` indicates that the intercluster terms are also present in the impurity solver as periodic boundary conditions of the cluster, so that they are subtracted as a momentum-independent real-space matrix.
"""
@inline Perturbation(intercluster::TBA; periodic::Bool=false) = CPTPerturbation(intercluster, periodic)
@inline function Perturbation(lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}; neighbors::Union{Int, Neighbors}=nneighbor(terms), periodic::Bool=false)
    return Perturbation(filter!(!isintracell, bonds(lattice, neighbors)), hilbert, terms; periodic)
end
function Perturbation(bonds::AbstractVector{<:Bond}, hilbert::Hilbert, terms::OneOrMore{Term}; periodic::Bool=false)
    return CPTPerturbation(TBA(Generator(bonds, hilbert, quadratic(OneOrMore(terms)); half=false)), periodic)
end

"""
    VCAPerturbation{V<:TBA, W<:TBA} <: Perturbation

VCA (Variational Cluster Approach) perturbation containing both intercluster quadratic terms and Weiss field terms of a system.

When `periodic` is `true`, the intercluster terms are also subtracted as a momentum-independent real-space matrix (like the Weiss terms), so that they can be included in the impurity solver as periodic boundary conditions of the cluster without changing the total Hamiltonian.
"""
struct VCAPerturbation{V<:TBA, W<:TBA} <: Perturbation
    intercluster::V
    weiss::W
    periodic::Bool
    function VCAPerturbation(intercluster::TBA, weiss::TBA, periodic::Bool=false)
        @assert kind(intercluster)==kind(weiss) "VCAPerturbation error: mismatched TBAKind for intercluster and weiss subsystem."
        new{typeof(intercluster), typeof(weiss)}(intercluster, weiss, periodic)
    end
end
@inline kind(::Type{<:VCAPerturbation{V}}) where {V<:TBA} = kind(V)
@inline function update!(perturbation::VCAPerturbation; parameters...)
    update!(perturbation.intercluster; parameters...)
    update!(perturbation.weiss; parameters...)
    return perturbation
end
@inline function (perturbation::VCAPerturbation)(k::Union{AbstractVector{<:Number}, Nothing}=nothing)
    result = matrix(perturbation.intercluster, k) - matrix(perturbation.weiss)
    perturbation.periodic && (result -= matrix(perturbation.intercluster))
    return result
end

"""
    Perturbation(intercluster::TBA, weiss::TBA; periodic::Bool=false) -> VCAPerturbation
    Perturbation(
        lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, weiss::OneOrMore{Term};
        neighbors::Union{Int, Neighbors}=max(nneighbor(terms), nneighbor(weiss)), periodic::Bool=false
    ) -> VCAPerturbation
    Perturbation(
        bonds₁::AbstractVector{<:Bond}, bonds₂::AbstractVector{<:Bond}, hilbert::Hilbert, terms::OneOrMore{Term}, weiss::OneOrMore{Term};
        periodic::Bool=false
    ) -> VCAPerturbation

Construct a VCA perturbation from pre-built `TBA` objects, or from lattice/hilbert/terms/weiss terms, or from separated bonds.

`periodic=true` indicates that the intercluster terms are also present in the impurity solver as periodic boundary conditions of the cluster, so that they are subtracted as a momentum-independent real-space matrix.
"""
@inline Perturbation(intercluster::TBA, weiss::TBA; periodic::Bool=false) = VCAPerturbation(intercluster, weiss, periodic)
@inline function Perturbation(
    lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, weiss::OneOrMore{Term};
    neighbors::Union{Int, Neighbors}=max(nneighbor(terms), nneighbor(weiss)), periodic::Bool=false
)
    bonds₀ = bonds(lattice, neighbors)
    bonds₁, bonds₂ = eltype(bonds₀)[], eltype(bonds₀)[]
    for bond in bonds₀
        isintracell(bond) ? push!(bonds₂, bond) : push!(bonds₁, bond)
    end
    return Perturbation(bonds₁, bonds₂, hilbert, terms, weiss; periodic)
end
function Perturbation(bonds₁::AbstractVector{<:Bond}, bonds₂::AbstractVector{<:Bond}, hilbert::Hilbert, terms::OneOrMore{Term}, weiss::OneOrMore{Term}; periodic::Bool=false)
    terms = quadratic(OneOrMore(terms))
    weiss = quadratic(OneOrMore(weiss))
    kind = TBAKind(typeof((terms..., weiss...)), valtype(hilbert))
    H = Generator(bonds₁, hilbert, terms; half=false)
    W = Generator(bonds₂, hilbert, weiss; half=false)
    return VCAPerturbation(TBA{typeof(kind)}(H), TBA{typeof(kind)}(W), periodic)
end

"""
    Periodization{N}

Structure for crystallographic periodization in quantum cluster theory.
Stores the coordinates of lattice operators and groups them by equivalence under lattice translations.
"""
struct Periodization{N}
    coordinates::Vector{SVector{N, Float64}}
    groups::Vector{Vector{Int}}
end

"""
    Periodization(
        ops_lattice::AbstractVector{<:CoordinatedIndex}, ops_unitcell::AbstractVector{<:CoordinatedIndex}, vectors::AbstractVector{<:AbstractVector{<:Number}};
        atol=atol, rtol=rtol
    ) -> Periodization

Construct a `Periodization` object by grouping lattice operators into translation-equivalent sets.
"""
function Periodization(ops_lattice::AbstractVector{<:CoordinatedIndex}, ops_unitcell::AbstractVector{<:CoordinatedIndex}, vectors::AbstractVector{<:AbstractVector{<:Number}}; atol=atol, rtol=rtol)
    coordinates = map(rcoordinate, ops_lattice)
    groups = Vector{Int}[]
    for op_unitcell in ops_unitcell
        group = Int[]
        for (i, op_lattice) in enumerate(ops_lattice)
            op_unitcell.index.internal == op_lattice.index.internal && issubordinate(rcoordinate(op_unitcell)-rcoordinate(op_lattice), vectors; atol, rtol) && push!(group, i)
        end
        push!(groups, group)
    end
    return Periodization(coordinates, groups)
end

"""
    (periodization::Periodization)(data::AbstractMatrix{<:Number}, k::AbstractVector{<:Number}) -> Matrix{ComplexF64}

Apply crystallographic periodization to data at a given crystal momentum `k`.
"""
function (periodization::Periodization)(data::AbstractMatrix{<:Number}, k::AbstractVector{<:Number})
    result = zeros(ComplexF64, length(periodization.groups), length(periodization.groups))
    for (i, groupᵢ) in enumerate(periodization.groups), (j, groupⱼ) in enumerate(periodization.groups)
        for m in groupᵢ, n in groupⱼ
            result[i, j] += data[m, n] * exp(1im*dot(k, periodization.coordinates[n]-periodization.coordinates[m]))
        end
    end
    L = count(periodization)
    for index in eachindex(result)
        result[index] /= L
    end
    return result
end

"""
    count(periodization::Periodization) -> Int

Return the number of unit cells in the cluster (i.e., the ratio of cluster size to unitcell size).
"""
@inline Base.count(periodization::Periodization) = length(periodization.coordinates)÷length(periodization.groups)

"""
    matrix(periodization::Periodization, k::AbstractVector{<:Number}) -> Matrix{ComplexF64}

The periodization as a linear map `U` such that the periodized data is `U*data*U'`.

Precomputing the map shares the momentum-dependent phase factors across repeated evaluations at the same momentum (e.g., over a frequency grid), turning each evaluation into two small matrix multiplications instead of recomputing the phases per call.
"""
function matrix(periodization::Periodization, k::AbstractVector{<:Number})
    L = count(periodization)
    result = zeros(ComplexF64, length(periodization.groups), length(periodization.coordinates))
    for (i, group) in enumerate(periodization.groups), m in group
        result[i, m] = exp(-1im*dot(k, periodization.coordinates[m]))/sqrt(L)
    end
    return result
end

"""
    QCT{U<:AbstractLattice, L<:AbstractLattice, I<:ImpuritySolver, V<:Perturbation, P<:Periodization} <: LatticeModel

Quantum cluster theory frontend that combines an impurity solver, perturbation, and crystallographic periodization to compute the Green's function of a system.
"""
struct QCT{U<:AbstractLattice, L<:AbstractLattice, I<:ImpuritySolver, V<:Perturbation, P<:Periodization} <: LatticeModel
    unitcell::U
    lattice::L
    solver::I
    perturbation::V
    periodization::P
end
@inline Parameters(qct::QCT) = Parameters(qct.solver)
@inline function update!(qct::QCT; parameters...)
    update!(qct.solver; parameters...)
    update!(qct.perturbation; parameters...)
    return qct
end

"""
    CPT{U<:AbstractLattice, L<:AbstractLattice, I<:ImpuritySolver, V<:CPTPerturbation, P<:Periodization}

Alias for [`QCT`](@ref) with `V<:CPTPerturbation`, representing cluster perturbation theory.
"""
const CPT{U<:AbstractLattice, L<:AbstractLattice, I<:ImpuritySolver, V<:CPTPerturbation, P<:Periodization} = QCT{U, L, I, V, P}

"""
    VCA{U<:AbstractLattice, L<:AbstractLattice, I<:ImpuritySolver, V<:VCAPerturbation, P<:Periodization}

Alias for [`QCT`](@ref) with `V<:VCAPerturbation`, representing the Variational Cluster Approach (VCA).
"""
const VCA{U<:AbstractLattice, L<:AbstractLattice, I<:ImpuritySolver, V<:VCAPerturbation, P<:Periodization} = QCT{U, L, I, V, P}

"""
    (qct::Union{QCT, Algorithm{<:QCT}})(ω::Number, k::Union{AbstractVector{<:Number}, Nothing}=nothing; periodization::Bool=true)

Evaluate the Green's function at frequency `ω` and momentum `k` by use of quantum cluster theory.
When `k` is `nothing`, no periodization is performed even if `periodization=true`.
"""
@inline (qct::Algorithm{<:QCT})(ω::Number, k::Union{AbstractVector{<:Number}, Nothing}=nothing; periodization::Bool=true) = (qct.frontend)(ω, k; periodization)
function (qct::QCT)(ω::Number, k::Union{AbstractVector{<:Number}, Nothing}=nothing; periodization::Bool=true)
    G⁻¹, V = inv(qct.solver, ω), qct.perturbation(k)
    result = inv(G⁻¹-V)
    !isnothing(k) && periodization && (result = qct.periodization(result, k))
    return result
end

"""
    TBA(qct::Union{QCT, Algorithm{<:QCT}}) -> TBA

The bare tight-binding system on the unitcell of a quantum cluster theory, reconstructed from the physical quadratic terms and the Hilbert space carried by its perturbation, so that consistency with the Green's function is guaranteed by construction. Its Bloch Hamiltonian is the bare band structure `h₀(k)`, and its velocity vertices (`velocity`) enter e.g. the `Conductivity` task.

The Weiss fields of VCA frontend are excluded. An error is thrown for a frontend constructed from pre-computed operators (e.g., from Wannier90), whose perturbation carries no terms.
"""
@inline TBA(qct::Algorithm{<:QCT}) = TBA(qct.frontend)
function TBA(qct::QCT)
    intercluster = qct.perturbation.intercluster
    isempty(intercluster.system.terms) && error("TBA error: the bare tight-binding system is unavailable because the quantum cluster frontend was constructed from pre-computed operators rather than terms.")
    hilbert = intercluster.system.hilbert
    return TBA{typeof(kind(intercluster))}(qct.unitcell, Hilbert([hilbert[site] for site in eachindex(qct.unitcell)]), intercluster.system.terms)
end

"""
    Ω(qct::Algorithm{<:QCT}; kwargs...) -> Float64
    Ω(
        qct::QCT;
        brillouinzone::BrillouinZone=BrillouinZone(qct.lattice, 100), μ::Real=0.0, atol::Real=1e-6, rtol::Real=1e-6, maxevals::Int=10^6
    ) -> Float64

Compute the grand potential per unit cell of the quantum cluster theory system.

For `Algorithm{<:VCA}`, this delegates to the second method.
"""
@inline Ω(qct::Algorithm{<:QCT}; kwargs...) = Ω(qct.frontend; kwargs...)
function Ω(qct::QCT; brillouinzone::BrillouinZone=BrillouinZone(qct.lattice, 100), μ::Real=0.0, atol::Real=1e-6, rtol::Real=1e-6, maxevals::Int=10^6)
    Vs = [qct.perturbation(k) for k in brillouinzone]
    buffer = Matrix{ComplexF64}(undef, size(first(Vs)))
    function f(ω::Real)
        result = 0.0
        G = qct.solver(1im*ω+μ)
        for V in Vs
            # in-place equivalent of `result -= log(abs(det(I-V*G)))`
            copyto!(buffer, I)
            mul!(buffer, V, G, -1, true)
            result -= logabsdet(lu!(buffer))[1]
        end
        return result / length(brillouinzone) / π
    end
    part₁ = quadgk(f, 0, Inf; atol, rtol, maxevals)[1]
    part₂ = real(mapreduce(tr, +, Vs)) / length(brillouinzone) / 2
    result = (Ω(qct.solver) + (part₁ + part₂)/factor(kind(qct.perturbation))) / count(qct.periodization)
    return result::Float64
end
@inline factor(::TBAKind{:TBA}) = 1
@inline factor(::TBAKind{:BdG}) = 2

"""
    expectation(qct::Algorithm{<:QCT}, m::Union{AbstractMatrix{<:Number}, Function, Symbol}; kwargs...) -> ComplexF64
    expectation(
        qct::QCT, m::Union{AbstractMatrix{<:Number}, Function, Symbol};
        brillouinzone::BrillouinZone=BrillouinZone(qct.lattice, 100), μ::Real=0.0, p::Real=1.0, atol::Real=1e-6, rtol::Real=1e-6, maxevals::Int=10^6
    ) -> ComplexF64

Compute the expectation value of an operator `m` (matrix, or function of `k`, or the symbol specifies the Weiss term in VCA) over the quantum cluster theory system by integrating over the Brillouin zone and frequency.

For `Algorithm{<:QCT}`, this delegates to the second method.
"""
@inline expectation(qct::Algorithm{<:QCT}, m::Union{AbstractMatrix{<:Number}, Function, Symbol}; kwargs...) = expectation(qct.frontend, m; kwargs...)
function expectation(qct::VCA, m::Symbol; kwargs...)
    weiss = update!(deepcopy(qct.perturbation.weiss); m=>1)
    selected = TBA{typeof(kind(weiss))}(Generator(expand(weiss.system, m)), weiss.quadraticization, weiss.commutator)
    f(k::AbstractVector{<:Number}) = matrix(selected, k).H.data
    return expectation(qct, f; kwargs...)
end
function expectation(
    qct::QCT, m::Union{AbstractMatrix{<:Number}, Function};
    brillouinzone::BrillouinZone=BrillouinZone(qct.lattice, 100), μ::Real=0.0, p::Real=1.0, atol::Real=1e-6, rtol::Real=1e-6, maxevals::Int=10^6
)
    Vs = [qct.perturbation(k) for k in brillouinzone]
    Ss = [_matrix_(m, k) for k in brillouinzone]
    Ts = [tr(S) for S in Ss]
    buffer = Matrix{ComplexF64}(undef, size(first(Vs)))
    function f(ω::Real)
        result = Complex(0.0)
        G⁻¹ = inv(qct.solver, 1im*ω+μ)
        for (V, S, T) in zip(Vs, Ss, Ts)
            # in-place equivalent of `result += tr(S*inv(G⁻¹-V)) - T/(1im*ω-p)`
            copyto!(buffer, G⁻¹)
            axpby!(-1, V, 1, buffer)
            result += trprod(S, inv!(lu!(buffer))) - T/(1im*ω-p)
        end
        return result / length(brillouinzone) / 2π
    end
    result = (quadgk(f, -Inf, 0; atol, rtol, maxevals)[1] + quadgk(f, 0, Inf; atol, rtol, maxevals)[1]) / length(qct.lattice)
    return result
end
@inline _matrix_(m::AbstractMatrix{<:Number}, ::AbstractVector{<:Number}) = m
@inline _matrix_(m::Function, k::AbstractVector{<:Number}) = m(k)::AbstractMatrix{<:Number}

"""
    filling(qct::Algorithm{<:QCT}; kwargs...) -> Float64
    filling(
        qct::QCT;
        brillouinzone::BrillouinZone=BrillouinZone(qct.lattice, 100), μ::Real=0.0, atol::Real=1e-6, rtol::Real=1e-6, maxevals::Int=10^6
    ) -> Float64

Compute the filling (particle number per lattice site, summed over spins and orbitals) of a quantum cluster theory system at the given chemical potential `μ`. For superconducting (BdG) systems, only the particle (annihilation) sector of the Nambu space is counted, so no double counting occurs.

The chemical potential enters by shifting the frequency argument of the impurity solver's Green's function, so it is *additional* to any chemical potential term already present in the Hamiltonian, and changing it does not require re-diagonalization.

Within the fixed quantum-number sector of the impurity solver, this shift is exact for normal systems: the particle number is a good quantum number, so `-μN` is a constant of each sector and only translates the excitation energies without touching the eigenstates. For superconducting (BdG) systems, however, the particle number fluctuates, and the shift amounts to a frozen-eigenstate approximation: the quasiparticle energies are translated while the condensate is kept fixed, although the true eigenstate would readjust with `μ`. In this case, the chemical potential should be put in the Hamiltonian (e.g. an `Onsite(:μ, ...)` term) and the impurity solver re-diagonalized by `update!` instead.
"""
@inline filling(qct::Algorithm{<:QCT}; kwargs...) = filling(qct.frontend; kwargs...)
function filling(qct::QCT; brillouinzone::BrillouinZone=BrillouinZone(qct.lattice, 100), μ::Real=0.0, atol::Real=1e-6, rtol::Real=1e-6, maxevals::Int=10^6)
    return real(expectation(qct, numbermatrix(qct); brillouinzone, μ, atol, rtol, maxevals))::Float64
end
# The single-particle number operator as a matrix in the Green's function representation: the basis is ordered by the quadraticization table of the perturbation, and for BdG the projector onto the particle (annihilation) sector of the Nambu space is used to avoid double counting.
function numbermatrix(qct::QCT)
    ops = operators(kind(qct.perturbation), qct.lattice, qct.perturbation.intercluster.system.hilbert, qct.perturbation.intercluster.quadraticization.table)
    return Diagonal(map(op->isannihilation(op.index.internal) ? 1.0 : 0.0, ops))
end

"""
    chemicalpotential(qct::Algorithm{<:QCT}, n::Real; kwargs...) -> Float64
    chemicalpotential(
        qct::QCT, n::Real;
        bracket::Union{Nothing, Tuple{Real, Real}}=nothing, xatol::Real=1e-6, xrtol::Real=1e-6,
        brillouinzone::BrillouinZone=BrillouinZone(qct.lattice, 100), atol::Real=1e-6, rtol::Real=1e-6, maxevals::Int=10^6
    ) -> Float64

Find the chemical potential `μ` at which the filling of a quantum cluster theory system is `n`, by a bracketed root search (Brent's method) on the monotone function `filling(qct; μ)-n`.

The returned chemical potential is *additional* to any chemical potential term already present in the Hamiltonian (see [`filling`](@ref)). Since the filling is monotone non-decreasing in `μ`, the search is globally convergent; when `n` sits on a gapped plateau, a value within the plateau is returned, which is physically correct as the chemical potential is not uniquely determined inside a gap. The frozen-eigenstate caveat of [`filling`](@ref) for BdG systems applies to the returned value as well. The bracket is found automatically by interval doubling unless explicitly provided. `xatol` and `xrtol` control the tolerance on `μ`, while the remaining keywords are passed to [`filling`](@ref).
"""
@inline chemicalpotential(qct::Algorithm{<:QCT}, n::Real; kwargs...) = chemicalpotential(qct.frontend, n; kwargs...)
function chemicalpotential(
    qct::QCT, n::Real;
    bracket::Union{Nothing, Tuple{Real, Real}}=nothing, xatol::Real=1e-6, xrtol::Real=1e-6,
    brillouinzone::BrillouinZone=BrillouinZone(qct.lattice, 100), atol::Real=1e-6, rtol::Real=1e-6, maxevals::Int=10^6
)
    nmax = tr(numbermatrix(qct))/length(qct.lattice)
    0<n<nmax || error("chemicalpotential error: target filling $n is out of the physical range (0, $nmax).")
    f(μ::Real) = filling(qct; μ=μ, brillouinzone, atol, rtol, maxevals)-n
    (lo, hi) = isnothing(bracket) ? _bracket_(f) : bracket
    lo==hi && return float(lo)
    return find_zero(f, (float(lo), float(hi)), Brent(); xatol=xatol, xrtol=xrtol)::Float64
end
# Bracket a sign change of the monotone non-decreasing function `f` by doubling an initial symmetric interval.
# A degenerate bracket `(μ, μ)` is returned when `f(μ)==0` is hit at an endpoint, which the caller handles.
function _bracket_(f::Function)
    lo, hi = -1.0, 1.0
    flo, fhi = f(lo), f(hi)
    for _ in 1:32
        flo==0 && return (lo, lo)
        fhi==0 && return (hi, hi)
        flo<0<fhi && return (lo, hi)
        if flo>0
            lo *= 2
            flo = f(lo)
        else
            hi *= 2
            fhi = f(hi)
        end
    end
    error("chemicalpotential error: failed to bracket the target filling automatically; please provide an explicit `bracket`.")
end

"""
    optimize!(vca::Algorithm{<:VCA}; kwargs...)
    optimize!(
        vca::VCA;
        verbose=false, method=LBFGS(), options=Options(x_abstol=1e-4, x_reltol=1e-4, f_abstol=2e-6, f_reltol=2e-6),
        Ω_options=(brillouinzone=BrillouinZone(vca.lattice, 100), μ=0.0, atol=1e-6, rtol=1e-6, maxevals=10^6)
    )

Optimize the variational cluster approximation to find the stationary point of the grand potential.

For `Algorithm{<:VCA}`, this delegates to the second method.
For `VCA`, the parameters are optimized using the specified method.
"""
@inline optimize!(vca::Algorithm{<:VCA}; kwargs...) = optimize!(vca.frontend; kwargs...)
function optimize!(
    vca::VCA;
    verbose=false, method=LBFGS(), options=Options(x_abstol=1e-4, x_reltol=1e-4, f_abstol=2e-6, f_reltol=2e-6),
    Ω_options=(brillouinzone=BrillouinZone(vca.lattice, 100), μ=0.0, atol=1e-6, rtol=1e-6, maxevals=10^6)
    )
    params = Parameters(vca.perturbation.weiss)
    function f(v::Vector{<:Number})
        parameters = Parameters{keys(params)}(v...)
        update!(vca; parameters...)
        verbose && println(parameters)
        return Ω(vca; Ω_options...)
    end
    op = optimize(f, collect(map(real, values(params))), method, options)
    parameters = Parameters{keys(params)}(op.minimizer...)
    update!(vca; parameters...)
    return vca, op
end

"""
    trprod(A::AbstractMatrix, B::AbstractMatrix) -> Number

Compute `tr(A*B)` without forming the product matrix.
"""
@inline function trprod(A::AbstractMatrix, B::AbstractMatrix)
    n = size(A, 1)
    @assert n==size(A, 2)==size(B, 1)==size(B, 2)
    s = zero(promote_type(eltype(A), eltype(B)))
    @inbounds for j in 1:n, i in 1:n
        s += A[i, j]*B[j, i]
    end
    return s
end

"""
    DynamicalSpectra{R<:ReciprocalSpace, F<:Function}

Dynamical spectra.
"""
struct DynamicalSpectra{R<:ReciprocalSpace, F<:Function}
    reciprocalspace::R
    energies::Vector{Float64}
    η::Float64
    rescale::F
    function DynamicalSpectra(reciprocalspace::ReciprocalSpace, energies::AbstractVector{<:Real}; η::Real=0.1, rescale::Function=identity)
        return new{typeof(reciprocalspace), typeof(rescale)}(reciprocalspace, collect(Float64, energies), convert(Float64, η), rescale)
    end
end
@inline dependencytypes(::Type{<:DynamicalSpectra}) = ()

"""
    DynamicalSpectraData{R<:ReciprocalSpace} <: Data

Data of dynamical spectra computed from quantum cluster theory, including:

1) `reciprocalspace::R`: reciprocal space on which the spectra are computed.
2) `energies::Vector{Float64}`: energy sample points.
3) `values::Matrix{Float64}`: spectral function A(ω,k) = -Im[Tr[G(ω,k)]] at each energy-momentum point.
"""
struct DynamicalSpectraData{R<:ReciprocalSpace} <: Data
    reciprocalspace::R
    energies::Vector{Float64}
    values::Matrix{Float64}
end
function run!(qct::Algorithm{<:QCT}, ds::Assignment{<:DynamicalSpectra})
    η, rescale = ds.task.η, ds.task.rescale
    result = zeros(length(ds.task.energies), length(ds.task.reciprocalspace))
    Vs = [qct.frontend.perturbation(k) for k in ds.task.reciprocalspace]
    # M = U'*U so that `tr(U*X*U') == tr(X*M)` by the cyclicity of trace, avoiding the U-sandwich per frequency
    Ms = [(U = matrix(qct.frontend.periodization, k); U'*U) for k in ds.task.reciprocalspace]
    buffer = Matrix{ComplexF64}(undef, size(first(Vs)))
    for (i, ω) in enumerate(ds.task.energies)
        G⁻¹ = inv(qct.frontend.solver, ω+1im*η)
        for (j, (V, M)) in enumerate(zip(Vs, Ms))
            # in-place equivalent of `result[i, j] = rescale(-imag(tr(U*inv(G⁻¹-V)*U')))`
            copyto!(buffer, G⁻¹)
            axpby!(-1, V, 1, buffer)
            result[i, j] = rescale(-imag(trprod(inv!(lu!(buffer)), M)))
        end
    end
    return DynamicalSpectraData(ds.task.reciprocalspace, ds.task.energies, result)
end
