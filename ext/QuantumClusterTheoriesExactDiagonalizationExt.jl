module QuantumClusterTheoriesExactDiagonalizationExt

using ExactDiagonalization: Abelian, BandLanczosMethod, ED, EDKind, EDMatrixization, RetardedGreenFunction, Sector
using QuantumLattices: AbstractLattice, Generator, Hilbert, Metric, Neighbors, OneOrMore, QuantumOperator, Table, Term, bonds, isintracell, kind, nneighbor, atol, eager, plain, rtol
using QuantumClusterTheories: Periodization, operators, perturbation, quadratic
using TightBindingApproximation: TBAKind
import QuantumClusterTheories: CPT, ImpuritySolver

"""
"""
mutable struct Cache
    ω::ComplexF64
    const data::Matrix{ComplexF64}
end

"""
"""
struct EDSolver{E<:ED, G<:RetardedGreenFunction} <: ImpuritySolver
    ed::E
    gf::G
    cache::Cache
    function EDSolver(ed::ED, gf::RetardedGreenFunction)
        new{typeof(ed), typeof(gf)}(ed, gf, Cache(zero(ComplexF64), gf(zero(ComplexF64))))
    end
end
@inline function (solver::EDSolver)(ω::Number)
    if ω≈solver.cache.ω
        return solver.cache.data
    else
        solver.cache.ω = ω
        return solver.gf(fill!(solver.cache.data, 0), ω)
    end
end

"""
"""
@inline function ImpuritySolver(ed::ED, operators::AbstractVector{<:QuantumOperator}, method=BandLanczosMethod(); kwargs...)
    return EDSolver(ed, RetardedGreenFunction(operators, ed, method; kwargs...))
end
function ImpuritySolver(
    lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian}, method=BandLanczosMethod(), dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms), kwargs...
)
    system = Generator(filter!(isintracell, bonds(lattice, neighbors)), hilbert, OneOrMore(terms), plain, eager; half=false)
    edkind = EDKind(hilbert)
    table = Table(hilbert, Metric(edkind, hilbert))
    sectors = broadcast(Sector, OneOrMore(quantumnumbers), hilbert; table)
    matrixization = EDMatrixization{dtype}(table, sectors...)
    ed = ED{typeof(edkind)}(lattice, system, matrixization)
    ops = operators(TBAKind(typeof(quadratic(terms)), valtype(hilbert)), lattice, hilbert)
    return ImpuritySolver(ed, ops, method; kwargs...)
end

"""
"""
function CPT(
    unitcell::AbstractLattice, lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian}, method=BandLanczosMethod(), dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms), atol=atol, rtol=rtol, kwargs...
)
    solver = ImpuritySolver(lattice, hilbert, terms, quantumnumbers, method, dtype; neighbors)
    pert = perturbation(lattice, hilbert, terms; neighbors)
    tbakind = kind(pert)
    ops_lattice = operators(tbakind, lattice, hilbert)
    ops_unitcell = operators(tbakind, unitcell, hilbert)
    periodization = Periodization(ops_lattice, ops_unitcell, unitcell.vectors)
    return CPT(unitcell, lattice, solver, pert, periodization)
end

end # module