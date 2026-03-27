module QuantumClusterTheories

using LinearAlgebra: dot, inv
using QuantumLattices: AbstractLattice, CoordinatedIndex, Fock, Frontend, Generator, Hilbert, Index, Metric, Neighbors, OneAtLeast, OneOrMore, Table, Term
using QuantumLattices: atol, bonds, isannihilation, isintracell, issubordinate, lazy, matrix, nneighbor, plain, rank, rcoordinate, rtol
using StaticArrays: SVector
using TightBindingApproximation: Quadraticization, TBA, TBAKind, commutator

export CPT, ImpuritySolver, Periodization, operators, perturbation, quadratic

"""
"""
abstract type ImpuritySolver end

"""
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
"""
function perturbation(lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}; neighbors::Union{Int, Neighbors}=nneighbor(terms))
    terms = quadratic(OneOrMore(terms))
    kind = TBAKind(typeof(terms), valtype(hilbert))
    H = Generator(filter!(!isintracell, bonds(lattice, neighbors)), hilbert, terms, plain, lazy; half=false)
    quadraticization = Quadraticization{typeof(kind)}(Table(hilbert, Metric(kind, hilbert)))
    commt = commutator(kind, hilbert)
    return TBA{typeof(kind)}(lattice, H, quadraticization, commt)
end
@generated quadratic(terms::OneAtLeast{Term}) = Expr(:tuple, [:(terms[$i]) for (i, T) in enumerate(fieldtypes(terms)) if rank(T)==2]...)

"""
"""
struct Periodization{N}
    coordinates::Vector{SVector{N, Float64}}
    groups::Vector{Vector{Int}}
end
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
"""
struct CPT{L<:AbstractLattice, I<:ImpuritySolver, T<:TBA, P<:Periodization} <: Frontend
    unitcell::L
    lattice::L
    solver::I
    perturbation::T
    periodization::P
end

"""
"""
@inline (cpt::CPT)(ω::Number) = inv(inv(cpt.solver(ω))-matrix(cpt.perturbation))
@inline (cpt::CPT)(ω::Number, k::AbstractVector{<:Number}) = cpt.periodization(inv(inv(cpt.solver(ω))-matrix(cpt.perturbation, k)), k)

end
