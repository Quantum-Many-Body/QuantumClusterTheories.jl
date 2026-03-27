module QuantumClusterTheoriesExactDiagonalizationExt

using ExactDiagonalization: Abelian, BandLanczosMethod, ED, EDKind, EDMatrixization, RetardedGreenFunction, Sector
using QuantumLattices: AbstractLattice, Generator, Hilbert, Metric, Neighbors, OneOrMore, QuantumOperator, Table, Term, bonds, isintracell, kind, nneighbor, atol, eager, plain, rtol
using QuantumClusterTheories: Periodization, operators, perturbation, quadratic
using TightBindingApproximation: TBAKind
import QuantumClusterTheories: CPT, ImpuritySolver

"""
"""
struct EDSolver{E<:ED, O<:QuantumOperator, G<:RetardedGreenFunction} <: ImpuritySolver
    ed::E
    operators::Vector{O}
    gf::G
end
@inline (solver::EDSolver)(ω::Number) = solver.gf(ω)

"""
"""
@inline function ImpuritySolver(ed::ED, operators::AbstractVector{<:QuantumOperator}, method=BandLanczosMethod(); kwargs...)
    return EDSolver(ed, operators, RetardedGreenFunction(operators, ed, method; kwargs...))
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
    tbakind = TBAKind(typeof(quadratic(terms)), valtype(hilbert))
    ops = operators(tbakind, lattice, hilbert, Table(hilbert, Metric(tbakind, hilbert)))
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
    ops_lattice = operators(tbakind, lattice, hilbert, Table(hilbert, Metric(tbakind, hilbert)))
    ops_unitcell = operators(tbakind, unitcell, hilbert, Table(hilbert, Metric(tbakind, hilbert)))
    periodization = Periodization(ops_lattice, ops_unitcell, unitcell.vectors)
    return CPT(unitcell, lattice, solver, pert, periodization)
end

end # module