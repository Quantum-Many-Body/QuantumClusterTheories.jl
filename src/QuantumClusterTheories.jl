module QuantumClusterTheories

using LinearAlgebra: inv
using QuantumLattices: AbstractLattice, Frontend, Generator, Hilbert, Metric, Neighbors, OneOrMore, Term, bonds, isintracell, issubordinate, lazy, matrix, nneighbor, plain
using StaticArrays: SVector
using TightBindingApproximation: Quadraticization, TBA, TBAKind

abstract type ImpuritySolver end

"""
"""
struct Periodization{N}
    coordinates::Vector{SVector{N, Float64}}
    groups::Vector{Vector{Int}}
end
function Periodization(coordinates::AbstractVector{<:AbstractVector{<:Number}}, vectors::AbstractVector{<:AbstractVector{<:Number}}; atol, rtol)
    groups = Vector{Int}[]
    indexes = collect(eachindex(coordinates))
    while !isempty(indexes)
        i = popfirst!(indexes)
        coordinateᵢ = coordinates[i]
        group = [i]
        for j in indexes
            coordinateⱼ = coordinates[j]
            issubordinate(coordinateᵢ-coordinateⱼ, vectors; atol, rtol) && push!(group, j)
        end
        filter!(∉(group), indexes)
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

function perturbation(lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}; neighbors::Union{Int, Neighbors}=nneighbor(terms))
    K = typeof(TBAKind(typeof(terms), valtype(hilbert)))
    H = Generator(filter!(!isintracell, bonds(lattice, neighbors)), hilbert, terms, plain, lazy; half=false)
    quadraticization = Quadraticization{K}(Table(hilbert, Metric(K(), hilbert)))
    commt = commutator(K(), hilbert)
    return TBA{K}(lattice, H, quadraticization, commt)
end

"""
"""
@inline (cpt::CPT)(ω::Number) = inv(inv(cpt.solver(ω))-matrix(cpt.perturbation))
@inline (cpt::CPT)(ω::Number, k::AbstractVector{<:Number}) = cpt.periodization(inv(inv(cpt.solver(ω))-matrix(cpt.perturbation, k)), k)

end
