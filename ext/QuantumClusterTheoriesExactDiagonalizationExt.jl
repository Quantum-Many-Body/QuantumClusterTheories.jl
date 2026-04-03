module QuantumClusterTheoriesExactDiagonalizationExt

using ExactDiagonalization: Abelian, BandLanczosMethod, ED, EDKind, EDMatrixization, GreenFunctionMethod, RetardedGreenFunction, Sector
using QuantumLattices: AbstractLattice, Generator, Hilbert, Lattice, Metric, Neighbors, OneAtLeast, OneOrMore, QuantumOperator, Table, Term, bonds, isintracell, kind, nneighbor, atol, eager, plain, rtol
using QuantumClusterTheories: Periodization, Perturbation, QCT, operators, qcttimer, quadratic
using TightBindingApproximation: TBAKind
using TimerOutputs: TimerOutput
import QuantumClusterTheories: CPT, ImpuritySolver
import QuantumLattices: Parameters, update!

"""
    Cache

Mutable cache for storing the retarded Green's function data at a given frequency.
"""
mutable struct Cache
    ω::ComplexF64
    const data::Matrix{ComplexF64}
end

"""
    EDSolver{E<:ED, G<:RetardedGreenFunction, O<:QuantumOperator, M<:GreenFunctionMethod} <: ImpuritySolver

Exact diagonalization based impurity solver computing the retarded Green's function with caching.
"""
mutable struct EDSolver{E<:ED, G<:RetardedGreenFunction, O<:QuantumOperator, M<:GreenFunctionMethod} <: ImpuritySolver
    const ed::E
    gf::G
    const operators::Vector{O}
    const method::M
    const cache::Cache
end
@inline function EDSolver(ed::ED, gf::RetardedGreenFunction, operators::AbstractVector{<:QuantumOperator}, method::GreenFunctionMethod)
    return EDSolver(ed, gf, operators, method, Cache(0im, gf(0im)))
end
@inline Parameters(solver::EDSolver) = Parameters(solver.ed)
@inline function update!(solver::EDSolver; timer::TimerOutput=qcttimer, kwargs...)
    update!(solver.ed; kwargs...)
    solver.gf = RetardedGreenFunction(solver.operators, solver.ed, solver.method; timer)
    solver.cache.ω = 0im
    solver.cache.data .= solver.gf(0im)
    return solver
end

"""
    (solver::EDSolver)(ω::Number) -> Matrix{ComplexF64}

Evaluate the retarded Green's function at frequency `ω` using cached results when available.
"""
@inline function (solver::EDSolver)(ω::Number)
    if ω≈solver.cache.ω
        return solver.cache.data
    else
        solver.cache.ω = ω
        return solver.gf(fill!(solver.cache.data, 0), ω)
    end
end

"""
    ImpuritySolver(
        lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian}, method=BandLanczosMethod(), dtype::Type{<:Number}=valtype(terms);
        neighbors::Union{Int, Neighbors}=nneighbor(terms), kwargs...
    ) -> EDSolver

Construct an exact diagonalization based impurity solver from a lattice, hilbert space, terms, and quantum numbers.
"""
function ImpuritySolver(
    lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian}, method=BandLanczosMethod(), dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms), timer::TimerOutput=qcttimer, kwargs...
)
    system = Generator(filter!(isintracell, bonds(lattice, neighbors)), hilbert, OneOrMore(terms), plain, eager; half=false)
    edkind = EDKind(hilbert)
    table = Table(hilbert, Metric(edkind, hilbert))
    sectors = broadcast(Sector, OneOrMore(quantumnumbers), hilbert; table)
    matrixization = EDMatrixization{dtype}(table, sectors...)
    ed = ED{typeof(edkind)}(lattice, system, matrixization)
    ops = operators(TBAKind(typeof(quadratic(terms)), valtype(hilbert)), lattice, hilbert)
    gf = RetardedGreenFunction(ops, ed, method; timer, kwargs...)
    return EDSolver(ed, gf, ops, method)
end

"""
    CPT(unitcell::AbstractLattice, lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian}, method=BandLanczosMethod(), dtype::Type{<:Number}=valtype(terms); neighbors::Union{Int, Neighbors}=nneighbor(terms), atol=atol, rtol=rtol, kwargs...) -> CPT

Construct a cluster perturbation theory (CPT) frontend using exact diagonalization as the impurity solver.
"""
function CPT(
    unitcell::AbstractLattice, lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian}, method=BandLanczosMethod(), dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms), atol=atol, rtol=rtol, timer::TimerOutput=qcttimer, kwargs...
)
    solver = ImpuritySolver(lattice, hilbert, terms, quantumnumbers, method, dtype; neighbors, timer)
    pert = Perturbation(lattice, hilbert, terms; neighbors)
    tbakind = kind(pert)
    opsₗ = operators(tbakind, lattice, hilbert)
    opsᵤ = operators(tbakind, unitcell, hilbert)
    periodization = Periodization(opsₗ, opsᵤ, unitcell.vectors)
    return QCT(unitcell, lattice, solver, pert, periodization)
end

"""
    ComposedEDSolver{E<:EDSolver} <: ImpuritySolver

Composed exact diagonalization solver for partitioned clusters, computing block-diagonal retarded Green's function with caching.
"""
struct ComposedEDSolver{E<:EDSolver} <: ImpuritySolver
    blocks::Vector{Int}
    representatives::Vector{E}
    permutation::Vector{Int}
    cache::Cache
end
function ComposedEDSolver(blocks::AbstractVector{Int}, representatives::AbstractVector{<:EDSolver}, permutation::AbstractVector{Int})
    ms = [representatives[block](0im) for block in blocks]
    result = blockdiag!(zeros(ComplexF64, mapreduce(size, .+, ms)), blocks, block::Int->ms[block], permutation)
    return ComposedEDSolver(blocks, representatives, permutation, Cache(0im, result))
end
function blockdiag!(dest::Matrix{ComplexF64}, blocks::Vector{Int}, matrix::Function, permutation::AbstractVector{Int})
    row, col = 1, 1
    for block in blocks
        m = matrix(block)
        inc_row, inc_col = size(m)
        dest[row:row+inc_row-1, col:col+inc_col-1] = m
        row += inc_row
        col += inc_col
    end
    dest[:] = @view dest[permutation, permutation]
    return dest
end
@inline Parameters(solver::ComposedEDSolver) = Parameters(first(solver.representatives))
@inline function update!(solver::ComposedEDSolver; timer::TimerOutput=qcttimer, kwargs...)
    for rep in solver.representatives
        update!(rep; timer, kwargs...)
    end
    solver.cache.ω = 0im
    solver.cache.data .= solver(0im)
    return solver
end

"""
    (solver::ComposedEDSolver)(ω::Number) -> Matrix{ComplexF64}

Evaluate the block-diagonal retarded Green's function at frequency `ω` using cached results when available.
"""
@inline function (solver::ComposedEDSolver)(ω::Number)
    if ω≈solver.cache.ω
        return solver.cache.data
    else
        solver.cache.ω = ω
        return blockdiag!(fill!(solver.cache.data, 0), solver.blocks, block::Int->solver.representatives[block](ω), solver.permutation)
    end
end

"""
    ImpuritySolver(
        lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, partition::OneOrMore{Pair}, method=BandLanczosMethod(), dtype::Type{<:Number}=valtype(terms);
        neighbors::Union{Int, Neighbors}=nneighbor(terms), kwargs...
    ) -> ComposedEDSolver

Construct an exact diagonalization based impurity solver for a partitioned lattice, where `partition` maps cluster indices to tuples of site clusters and quantum numbers.
"""
function ImpuritySolver(
    lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, partition::OneOrMore{Pair}, method=BandLanczosMethod(), dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms), timer::TimerOutput=qcttimer, kwargs...
)
    blocks = Int[]
    sites = Int[]
    for (block, (clusters, _)) in enumerate(OneOrMore(partition))
        for cluster in OneOrMore(clusters)
            push!(blocks, block)
            append!(sites, cluster)
        end
    end
    sites = sortperm(sites)
    neighbors = isa(neighbors, Int) ? Neighbors(lattice, neighbors) : neighbors
    representatives = [
        begin
            subsites = first(OneOrMore(clusters))
            sublattice = Lattice(map(site->lattice[site], subsites)...)
            subhilbert = Hilbert([hilbert[site] for site in subsites])
            ImpuritySolver(sublattice, subhilbert, terms, quantumnumbers, method, dtype; neighbors, timer, kwargs...)
        end
        for (clusters, quantumnumbers) in OneOrMore(partition)
    ]
    permutation = invperm(sortperm(
        operators(TBAKind(typeof(quadratic(terms)), valtype(hilbert)), lattice, hilbert);
        by=op->sites[op.index.site]
    ))
    return ComposedEDSolver(blocks, representatives, permutation)
end

"""
    CPT(
        unitcell::AbstractLattice, lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, partition::OneOrMore{Pair}, method=BandLanczosMethod(), dtype::Type{<:Number}=valtype(terms);
        neighbors::Union{Int, Neighbors}=nneighbor(terms), kwargs...
    ) -> CPT

Construct a cluster perturbation theory (CPT) frontend using exact diagonalization as the impurity solver with lattice partitioning.
"""
function CPT(
    unitcell::AbstractLattice, lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, partition::OneOrMore{Pair}, method=BandLanczosMethod(), dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms), timer::TimerOutput=qcttimer, kwargs...
)
    solver = ImpuritySolver(lattice, hilbert, terms, partition, method, dtype; neighbors, timer)
    table = zeros(Int, length(lattice))
    num = 1
    for (clusters, _) in OneOrMore(partition)
        for cluster in OneOrMore(clusters)
            for site in cluster
                table[site] = num
            end
            num += 1
        end
    end
    pert = Perturbation(
        filter!(bonds(lattice, neighbors)) do bond
            isintracell(bond) || return true
            length(bond)==2 && return table[bond[1].site] ≠ table[bond[2].site]
            return false
        end,
        hilbert, terms
    )
    tbakind = kind(pert)
    opsₗ = operators(tbakind, lattice, hilbert)
    opsᵤ = operators(tbakind, unitcell, hilbert)
    periodization = Periodization(opsₗ, opsᵤ, unitcell.vectors)
    return QCT(unitcell, lattice, solver, pert, periodization)
end

end # module
