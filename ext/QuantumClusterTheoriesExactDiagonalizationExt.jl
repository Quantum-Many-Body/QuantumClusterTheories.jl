module QuantumClusterTheoriesExactDiagonalizationExt

using ExactDiagonalization: Abelian, BandLanczosMethod, ED, EDKind, EDMatrixization, GreenFunctionMethod, RetardedGreenFunction, Sector, normalize
using LinearAlgebra: eigen
using QuantumLattices: AbstractLattice, Generator, Hilbert, Lattice, Metric, Neighbors, OneAtLeast, OneOrMore, QuantumOperator, Table, Term, bonds, isintracell, kind, nneighbor
using QuantumClusterTheories: Periodization, Perturbation, QCT, operators, qcttimer, quadratic
using TightBindingApproximation: TBAKind
using TimerOutputs: TimerOutput
import QuantumClusterTheories: CPT, ImpuritySolver, VCA, Ω
import QuantumLattices: Parameters, contenttocache, contenttoconfig, qlcsave, qlload, stamp, update!

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
mutable struct EDSolver{E<:ED, O<:QuantumOperator, M<:GreenFunctionMethod, G<:RetardedGreenFunction} <: ImpuritySolver
    const ed::E
    const operators::Vector{O}
    const method::M
    const timer::TimerOutput
    Ω::Float64
    gf::G
    G::Cache
    G⁻¹::Cache
    function EDSolver(ed::ED, operators::AbstractVector{<:QuantumOperator}, method::GreenFunctionMethod; timer::TimerOutput=qcttimer)
        G = Core.Compiler.return_type(RetardedGreenFunction, Tuple{typeof(operators), typeof(ed), typeof(method)})
        return new{typeof(ed), eltype(operators), typeof(method), G}(ed, operators, method, timer)
    end
end
@inline Parameters(solver::EDSolver) = Parameters(solver.ed)
@inline contenttocache(solver::EDSolver) = (Ω=solver.Ω, gf=solver.gf)
@inline contenttoconfig(solver::EDSolver) = (contenttoconfig(solver.ed), solver.operators, solver.method)
function set!(solver::EDSolver; ω::Number=1e-4im)
    cached = try
        qlload(pathof(solver, :cache), stamp(solver))
    catch
        nothing
    end
    if isnothing(cached)
        eigensystem = eigen(solver.ed; nev=1, timer=solver.timer)
        solver.Ω, v₀, sector₀ = only(eigensystem.values), only(eigensystem.vectors), only(eigensystem.sectors)
        solver.gf = RetardedGreenFunction(solver.operators, solver.ed, solver.method; e₀=solver.Ω, v₀=v₀, sector₀=sector₀, timer=solver.timer)
        qlcsave(solver)
    else
        solver.Ω = cached.Ω
        solver.gf = cached.gf
    end
    solver.G = Cache(ω, solver.gf(ω))
    solver.G⁻¹ = Cache(ω, inv(solver.G.data))
    return solver
end
@inline function update!(solver::EDSolver; parameters...)
    if length(parameters) > 0
        update!(solver.ed; parameters...)
        set!(solver)
    end
    return solver
end

"""
    (solver::EDSolver)(ω::Number) -> Matrix{ComplexF64}

Evaluate the retarded Green's function at frequency `ω` using cached results when available.
"""
@inline function (solver::EDSolver)(ω::Number)
    if !isdefined(solver, :gf)
        set!(solver; ω=ω)
    elseif !(ω≈solver.G.ω)
        solver.G.ω = ω
        solver.gf(fill!(solver.G.data, 0), ω)
    end
    return solver.G.data
end

"""
    inv(solver::EDSolver, ω::Number) -> Matrix{ComplexF64}

Return the inverse of the retarded Green's function at frequency `ω` using cached results when available.
"""
@inline function Base.inv(solver::EDSolver, ω::Number)
    if !isdefined(solver, :gf)
        set!(solver; ω=ω)
    elseif !(ω≈solver.G⁻¹.ω)
        solver.G⁻¹ = Cache(ω, inv(solver(ω)))
    end
    return solver.G⁻¹.data
end

"""
    Ω(solver::EDSolver) -> Float64

Return the grand potential of the exact diagonalization solver.
"""
@inline Ω(solver::EDSolver) = solver.Ω

"""
    ImpuritySolver(
        lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian}, method=BandLanczosMethod(), dtype::Type{<:Number}=valtype(terms);
        neighbors::Union{Int, Neighbors}=nneighbor(terms), timer::TimerOutput=qcttimer
    ) -> EDSolver

Construct an exact diagonalization based impurity solver from a lattice, hilbert space, terms, and quantum numbers.
"""
function ImpuritySolver(
    lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian}, method=BandLanczosMethod(), dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms), timer::TimerOutput=qcttimer
)
    system = Generator(filter!(isintracell, bonds(lattice, neighbors)), hilbert, normalize(terms); half=false)
    edkind = EDKind(hilbert)
    table = Table(hilbert, Metric(edkind, hilbert))
    sectors = broadcast(Sector, OneOrMore(quantumnumbers), hilbert; table)
    matrixization = EDMatrixization{dtype}(table, sectors...)
    ed = ED{typeof(edkind)}(lattice, system, matrixization)
    ops = operators(TBAKind(typeof(quadratic(terms)), valtype(hilbert)), lattice, hilbert)
    return EDSolver(ed, ops, method; timer)
end

"""
    CPT(
        unitcell::AbstractLattice, lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian}, method=BandLanczosMethod(), dtype::Type{<:Number}=valtype(terms);
        neighbors::Union{Int, Neighbors}=nneighbor(terms), timer::TimerOutput=qcttimer
    ) -> CPT

Construct a cluster perturbation theory (CPT) frontend using exact diagonalization as the impurity solver.
"""
function CPT(
    unitcell::AbstractLattice, lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian}, method=BandLanczosMethod(), dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms), timer::TimerOutput=qcttimer
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
    VCA(
        unitcell::AbstractLattice, lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, weiss::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian},
        method=BandLanczosMethod(), dtype::Type{<:Number}=promote_type(valtype(terms), valtype(weiss));
        neighbors::Union{Int, Neighbors}=max(nneighbor(terms), nneighbor(weiss)), timer::TimerOutput=qcttimer
    ) -> VCA

Construct a Variational Cluster Approach (VCA) frontend using exact diagonalization as the impurity solver.
"""
function VCA(
    unitcell::AbstractLattice, lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, weiss::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian},
    method=BandLanczosMethod(), dtype::Type{<:Number}=promote_type(valtype(terms), valtype(weiss));
    neighbors::Union{Int, Neighbors}=max(nneighbor(terms), nneighbor(weiss)), timer::TimerOutput=qcttimer
)
    solver = ImpuritySolver(lattice, hilbert, (OneOrMore(terms)..., OneOrMore(weiss)...), quantumnumbers, method, dtype; neighbors, timer)
    pert = Perturbation(lattice, hilbert, terms, weiss; neighbors)
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
mutable struct ComposedEDSolver{E<:EDSolver} <: ImpuritySolver
    const blocks::Vector{Int}
    const representatives::Vector{E}
    const permutation::Vector{Int}
    G::Cache
    G⁻¹::Cache
    function ComposedEDSolver(blocks::AbstractVector{Int}, representatives::AbstractVector{<:EDSolver}, permutation::AbstractVector{Int})
        return new{eltype(representatives)}(blocks, representatives, permutation)
    end
end
function setG!(solver::ComposedEDSolver; ω::Number=1e-4im)
    ms = [solver.representatives[block](ω) for block in solver.blocks]
    if isdefined(solver, :G)
        fill!(solver.G.data, 0)
    else
        solver.G = Cache(ω, zeros(ComplexF64, mapreduce(size, .+, ms)))
    end
    solver.G.ω = ω
    row, col = 1, 1
    for block in solver.blocks
        m = ms[block]
        inc_row, inc_col = size(m)
        solver.G.data[row:row+inc_row-1, col:col+inc_col-1] = m
        row += inc_row
        col += inc_col
    end
    solver.G.data[:] = @view solver.G.data[solver.permutation, solver.permutation]
    return solver
end
function setG⁻¹!(solver::ComposedEDSolver; ω::Number=1e-4im)
    ms = [inv(solver.representatives[block], ω) for block in solver.blocks]
    if isdefined(solver, :G⁻¹)
        fill!(solver.G⁻¹.data, 0)
    else
        solver.G⁻¹ = Cache(ω, zeros(ComplexF64, mapreduce(size, .+, ms)))
    end
    solver.G⁻¹.ω = ω
    row, col = 1, 1
    for block in solver.blocks
        m = ms[block]
        inc_row, inc_col = size(m)
        solver.G⁻¹.data[row:row+inc_row-1, col:col+inc_col-1] = m
        row += inc_row
        col += inc_col
    end
    solver.G⁻¹.data[:] = @view solver.G⁻¹.data[solver.permutation, solver.permutation]
    return solver
end
@inline Parameters(solver::ComposedEDSolver) = mapreduce(Parameters, merge, solver.representatives)
@inline function update!(solver::ComposedEDSolver; parameters...)
    if length(parameters) > 0
        for rep in solver.representatives
            update!(rep; parameters...)
        end
        setG!(solver)
        setG⁻¹!(solver)
    end
    return solver
end

"""
    (solver::ComposedEDSolver)(ω::Number) -> Matrix{ComplexF64}

Evaluate the block-diagonal retarded Green's function at frequency `ω` using cached results when available.
"""
@inline function (solver::ComposedEDSolver)(ω::Number)
    (isdefined(solver, :G) && ω≈solver.G.ω) || setG!(solver; ω=ω)
    return solver.G.data
end

"""
    inv(solver::ComposedEDSolver, ω::Number) -> Matrix{ComplexF64}

Return the inverse of the block-diagonal retarded Green's function at frequency `ω` using cached results when available.
"""
function Base.inv(solver::ComposedEDSolver, ω::Number)
    (isdefined(solver, :G⁻¹) && ω≈solver.G⁻¹.ω) || setG⁻¹!(solver; ω=ω)
    return solver.G⁻¹.data
end

"""
    Ω(solver::ComposedEDSolver) -> Float64

Return the grand potential of the composed exact diagonalization solver.
"""
function Ω(solver::ComposedEDSolver)
    result = 0.0
    for block in solver.blocks
        result += solver.representatives[block].Ω
    end
    return result
end

"""
    ImpuritySolver(
        lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, partition::OneOrMore{Pair}, method=BandLanczosMethod(), dtype::Type{<:Number}=valtype(terms);
        neighbors::Union{Int, Neighbors}=nneighbor(terms), timer::TimerOutput=qcttimer
    ) -> ComposedEDSolver

Construct an exact diagonalization based impurity solver for a partitioned lattice, where `partition` maps cluster indices to tuples of site clusters and quantum numbers.
"""
function ImpuritySolver(
    lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, partition::OneOrMore{Pair}, method=BandLanczosMethod(), dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms), timer::TimerOutput=qcttimer
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
            ImpuritySolver(sublattice, subhilbert, terms, quantumnumbers, method, dtype; neighbors, timer)
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
        neighbors::Union{Int, Neighbors}=nneighbor(terms), timer::TimerOutput=qcttimer
    ) -> CPT

Construct a cluster perturbation theory (CPT) frontend using exact diagonalization as the impurity solver with lattice partitioning.
"""
function CPT(
    unitcell::AbstractLattice, lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, partition::OneOrMore{Pair}, method=BandLanczosMethod(), dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms), timer::TimerOutput=qcttimer
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
