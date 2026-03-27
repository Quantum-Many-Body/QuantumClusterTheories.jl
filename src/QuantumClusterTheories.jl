module QuantumClusterTheories

using LinearAlgebra: inv
using QuantumLattices: AbstractLattice, Fock, Frontend, Generator, Hilbert, Metric, Neighbors, OneAtLeast, OneOrMore, Table, Term
using QuantumLattices: atol, bonds, isannihilation, isintracell, issubordinate, lazy, matrix, nneighbor, plain, rank, rtol
using StaticArrays: SVector
using TightBindingApproximation: Quadraticization, TBA, TBAKind

"""
"""
abstract type ImpuritySolver end

"""
"""
struct Periodization{N}
    coordinates::Vector{SVector{N, Float64}}
    groups::Vector{Vector{Int}}
end
function Periodization(coordinates::AbstractVector{<:AbstractVector{<:Number}}, vectors::AbstractVector{<:AbstractVector{<:Number}}; atol=atol, rtol=rtol)
    groups = Vector{Int}[]
    indexes = collect(eachindex(coordinates))
    while !isempty(indexes)
        i = popfirst!(indexes)
        coordinateᵢ = coordinates[i]
        group = [i]
        if length(vectors)>0
            for j in indexes
                coordinateⱼ = coordinates[j]
                issubordinate(coordinateᵢ-coordinateⱼ, vectors; atol, rtol) && push!(group, j)
            end
            filter!(∉(group), indexes)
        end
        push!(groups, group)
    end
    return Periodization(coordinates, groups)
end

"""
"""
function (periodization::Periodization)(data::AbstractMatrix{<:Number}, k::AbstractVector{<:Number})
    N = length(periodization.groups)
    L = length(coordinates) ÷ N
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

"""
"""
function operators(::TBAKind{:TBA}, lattice::AbstractLattice, hilbert::Hilbert{<:Fock}, table::Table)
    result = [CoordinatedIndex(Index(site, fockindex), coordinate, zero(coordinate)) for (site, coordinate) in enumerate(lattice) for fockindex in hilbert[site] if isannihilation(fockindex)]
    return sort!(result; by=index->table[index])
end
function operators(::TBAKind{:BdG}, lattice::AbstractLattice, hilbert::Hilbert{<:Fock}, table::Table)
    result = [CoordinatedIndex(Index(site, fockindex), coordinate, zero(coordinate)) for (site, coordinate) in enumerate(lattice) for fockindex in hilbert[site]]
    return sort!(result; by=index->table[index])
end

"""
"""
function perturbation(lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}; neighbors::Union{Int, Neighbors}=nneighbor(terms))
    terms = quadratic(OneOrMore(terms))
    K = typeof(TBAKind(typeof(terms), valtype(hilbert)))
    H = Generator(filter!(!isintracell, bonds(lattice, neighbors)), hilbert, terms, plain, lazy; half=false)
    quadraticization = Quadraticization{K}(Table(hilbert, Metric(K(), hilbert)))
    commt = commutator(K(), hilbert)
    return TBA{K}(lattice, H, quadraticization, commt)
end
@generated quadratic(terms::OneAtLeast{Term}) = Expr(:tuple, [:(terms[$i]) for (i, T) in enumerate(fieldtypes(terms)) if rank(T)==2]...)

end
