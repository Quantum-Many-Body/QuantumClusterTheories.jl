module QuantumClusterTheories

using LinearAlgebra: dot, inv
using QuantumLattices: AbstractLattice, CoordinatedIndex, Fock, Frontend, Generator, Hilbert, Index, Metric, Neighbors, OneAtLeast, OneOrMore, Table, Term
using QuantumLattices: atol, bonds, isannihilation, isintracell, issubordinate, lazy, matrix, nneighbor, plain, rank, rcoordinate, rtol
using StaticArrays: SVector
using TightBindingApproximation: Quadraticization, TBA, TBAKind, commutator

export CPT, ImpuritySolver, Periodization, operators, perturbation, quadratic

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

Construct a tight-binding approximation (TBA) object by keeping only the quadratic (pairwise) interaction terms on inter-cellular bonds.
"""
function perturbation(lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}; neighbors::Union{Int, Neighbors}=nneighbor(terms))
    terms = quadratic(OneOrMore(terms))
    kind = TBAKind(typeof(terms), valtype(hilbert))
    H = Generator(filter!(!isintracell, bonds(lattice, neighbors)), hilbert, terms, plain, lazy; half=false)
    quadraticization = Quadraticization{typeof(kind)}(Table(hilbert, Metric(kind, hilbert)))
    commt = commutator(kind, hilbert)
    return TBA{typeof(kind)}(lattice, H, quadraticization, commt)
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
    CPT{L<:AbstractLattice, I<:ImpuritySolver, T<:TBA, P<:Periodization} <: Frontend

Cluster perturbation theory (CPT) frontend combining a unit cell, full lattice, impurity solver, perturbation, and periodization.
"""
struct CPT{L<:AbstractLattice, I<:ImpuritySolver, T<:TBA, P<:Periodization} <: Frontend
    unitcell::L
    lattice::L
    solver::I
    perturbation::T
    periodization::P
end

"""
    (cpt::CPT)(ω::Number) -> Matrix{ComplexF64}
    (cpt::CPT)(ω::Number, k::AbstractVector{<:Number}) -> Matrix{ComplexF64}

Evaluate the cluster perturbation theory single-particle Green's function.
With only `ω`, returns the real-space CPT Green's function. With `k`, returns the momentum-space periodised Green's function.
"""
@inline (cpt::CPT)(ω::Number) = inv(inv(cpt.solver(ω))-matrix(cpt.perturbation))
@inline (cpt::CPT)(ω::Number, k::AbstractVector{<:Number}) = cpt.periodization(inv(inv(cpt.solver(ω))-matrix(cpt.perturbation, k)), k)

end
