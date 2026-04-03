module QuantumClusterTheories

using LinearAlgebra: I, dot, inv, tr
using TimerOutputs: TimerOutput
using QuantumLattices: AbstractLattice, Action, Algorithm, Assignment, Bond, CoordinatedIndex, Data, Fock, Frontend, Generator, Hilbert, Index, Metric, Neighbors, OneAtLeast, OneOrMore, ReciprocalSpace, Table, Term
using QuantumLattices: atol, bonds, getcontent, isannihilation, isintracell, issubordinate, lazy, matrix, nneighbor, plain, rank, rcoordinate, rtol, update
using StaticArrays: SVector
using TightBindingApproximation: Quadraticization, TBA, TBAKind, commutator
import QuantumLattices: Parameters, options, run!, update!

export CPT, DynamicalSpectra, DynamicalSpectraData, ImpuritySolver, Periodization, QCT, QuantumClusterTheory, operators, perturbation, quadratic, qcttimer

"""
    const qcttimer

Default shared timer for all quantum cluster theory methods.
"""
const qcttimer = TimerOutput()

"""
    ImpuritySolver

Abstract type for impurity solvers used in quantum cluster theory calculations.
Subtypes must implement the call syntax `solver(ω)` to return the solver's response function at frequency `ω`.
"""
abstract type ImpuritySolver end

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
    perturbation(lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}; neighbors::Union{Int, Neighbors}=nneighbor(terms)) -> TBA
    perturbation(bonds::AbstractVector{<:Bond}, hilbert::Hilbert, terms::OneOrMore{Term}) -> TBA

Construct a tight-binding approximation (TBA) object by keeping only the quadratic (pairwise) interaction terms.

The first method extracts inter-cellular bonds from the lattice and delegates to the second method.
The second method takes bonds directly and constructs the TBA from the given `bonds`, `hilbert` space, and `terms`.
"""
@inline function perturbation(lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}; neighbors::Union{Int, Neighbors}=nneighbor(terms))
    return perturbation(filter!(!isintracell, bonds(lattice, neighbors)), hilbert, terms)
end
function perturbation(bonds::AbstractVector{<:Bond}, hilbert::Hilbert, terms::OneOrMore{Term})
    terms = quadratic(OneOrMore(terms))
    kind = TBAKind(typeof(terms), valtype(hilbert))
    H = Generator(bonds, hilbert, terms, plain, lazy; half=false)
    quadraticization = Quadraticization{typeof(kind)}(Table(hilbert, Metric(kind, hilbert)))
    commt = commutator(kind, hilbert)
    return TBA{typeof(kind)}(H, quadraticization, commt)
end

"""
    quadratic(terms::OneAtLeast{Term}) -> Tuple

Extract the quadratic (rank-2) terms from a collection of terms.
"""
@generated quadratic(terms::OneAtLeast{Term}) = Expr(:tuple, [:(terms[$i]) for (i, T) in enumerate(fieldtypes(terms)) if rank(T)==2]...)

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
    Periodization(ops_lattice, ops_unitcell, vectors; atol=atol, rtol=rtol) -> Periodization

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
    N = length(periodization.groups)
    L = length(periodization.coordinates) ÷ N
    result = zeros(ComplexF64, N, N)
    for (i, groupᵢ) in enumerate(periodization.groups), (j, groupⱼ) in enumerate(periodization.groups)
        for m in groupᵢ, n in groupⱼ
            result[i, j] += data[m, n] * exp(1im*dot(k, periodization.coordinates[n]-periodization.coordinates[m]))
        end
    end
    for index in eachindex(result)
        result[index] /= L
    end
    return result
end

"""
    abstract type QuantumClusterTheory <: Frontend end

Abstract base type for quantum cluster theories, which provide a framework for solving quantum many-body problems using cluster-based approximations.
All concrete quantum cluster theory implementations should subtype this abstract type.
"""
abstract type QuantumClusterTheory <: Frontend end
const QCT = QuantumClusterTheory
@inline Parameters(qct::QuantumClusterTheory) = Parameters(getcontent(qct, :solver))
@inline function update!(qct::Algorithm{<:QuantumClusterTheory}; kwargs...)
    if length(kwargs)>0
        qct.parameters = update(qct.parameters; kwargs...)
        update!(qct.frontend; timer=qct.timer, qct.map(qct.parameters)...)
    end
    return qct
end

"""
    CPT{U<:AbstractLattice, L<:AbstractLattice, I<:ImpuritySolver, T<:TBA, P<:Periodization} <: QuantumClusterTheory

Cluster perturbation theory (CPT) frontend combining a unit cell, full lattice, impurity solver, perturbation, and periodization.
"""
struct CPT{U<:AbstractLattice, L<:AbstractLattice, I<:ImpuritySolver, T<:TBA, P<:Periodization} <: QuantumClusterTheory
    unitcell::U
    lattice::L
    solver::I
    perturbation::T
    periodization::P
end
@inline function update!(cpt::CPT; timer::TimerOutput=qcttimer, kwargs...)
    update!(cpt.solver; timer, kwargs...)
    update!(cpt.perturbation; kwargs...)
    return cpt
end

"""
    (cpt::Union{CPT, Algorithm{<:CPT}})(ω::Number, k::Union{AbstractVector{<:Number}, Nothing}=nothing; periodization::Bool=true)

Evaluate the Cluster Perturbation Theory Green's function at frequency `ω` and momentum `k`.
When `k` is `nothing`, no periodization is performed even if `periodization=true`.
"""
@inline (cpt::Algorithm{<:CPT})(ω::Number, k::Union{AbstractVector{<:Number}, Nothing}=nothing; periodization::Bool=true) = (cpt.frontend)(ω, k; periodization)
function (cpt::CPT)(ω::Number, k::Union{AbstractVector{<:Number}, Nothing}=nothing; periodization::Bool=true)
    G, V = cpt.solver(ω), matrix(cpt.perturbation, k)
    result = G / (I-V*G)
    !isnothing(k) && periodization && (result = cpt.periodization(result, k))
    return result
end

"""
    DynamicalSpectra{R<:ReciprocalSpace} <: Action

Dynamical spectra.
"""
struct DynamicalSpectra{R<:ReciprocalSpace} <: Action
    reciprocalspace::R
    energies::Vector{Float64}
    DynamicalSpectra(reciprocalspace::ReciprocalSpace, energies::AbstractVector{<:Real}) = new{typeof(reciprocalspace)}(reciprocalspace, energies)
end
@inline options(::Type{<:Assignment{<:DynamicalSpectra}}) = (
    η = "Lorentz broadening",
    rescale = "function used to rescale the intensity of the spectrum at each energy-momentum point"
)

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
function run!(qct::Algorithm{<:QuantumClusterTheory}, ds::Assignment{<:DynamicalSpectra}; η::Real=0.1, rescale::Function=identity, options...)
    result = zeros(length(ds.action.energies), length(ds.action.reciprocalspace))
    for (i, ω) in enumerate(ds.action.energies)
        for (j, k) in enumerate(ds.action.reciprocalspace)
            result[i, j] = rescale(-imag(tr(qct(ω+1im*η, k))))
        end
    end
    return DynamicalSpectraData(ds.action.reciprocalspace, ds.action.energies, result)
end

end
