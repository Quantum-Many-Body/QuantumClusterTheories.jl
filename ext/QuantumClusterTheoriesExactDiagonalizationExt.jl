module QuantumClusterTheoriesExactDiagonalizationExt

using ExactDiagonalization: Abelian, BandLanczosMethod, ED, EDKind, EDMatrixization, RetardedGreenFunction
using QuantumLattices: AbstractLattice, Hilbert, Metric, Neighbors, OneOrMore, QuantumOperator, Table, Term, nneighbor, atol, eager, plain, rtol
using QuantumClusterTheories: ImpuritySolver, Periodization, operators, perturbation, quadratic 
using TightBindingApproximation: TBAKind

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
@inline function EDSolver(ed::ED, operators::AbstractVector{<:QuantumOperator}, method=BandLanczosMethod(); kwargs...)
    return EDSolver(ed, operators, RetardedGreenFunction(operators, ed, method; kwargs...))
end
function EDSolver(
    lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian}, method=BandLanczosMethod(), dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms), kwargs...
)
    system = Generator(filter!(isintracell, bonds(lattice, neighbors)), hilbert, OneOrMore(terms), plain, eager; half=false)
    edkind = EDKind(hilbert)
    table = Table(hilbert, Metric(edkind, hilbert))
    sectors = broadcast(Sector, OneOrMore(quantumnumbers), hilbert; table)
    matrixization = EDMatrixization{dtype}(table, sectors...)
    ed = ED{typeof(edkind)}(lattice, system, matrixization)
    tbakind = TBAKind(typeof(terms), valtype(hilbert))
    ops = operators(tbakind, lattice, hilbert, Table(hilbert, Metric(tbakind, hilbert)))
    return EDSolver(ed, ops, method; kwargs...)
end

"""
"""
function CPT(
    unitcell::AbstractLattice, lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, quantumnumbers::OneOrMore{Abelian}, method=BandLanczosMethod(), dtype::Type{<:Number}=valtype(terms);
    neighbors::Union{Int, Neighbors}=nneighbor(terms), atol=atol, rtol=rtol, kwargs...
)
    solver = EDSolver(lattice, hilbert, terms, quantumnumbers, method, dtype; neighbors)
    pert = perturbation(lattice, hilbert, terms; neighbors)
    periodization = Periodization([rcoordinate(op) for op in solver.operators], unitcell.vectors; atol, rtol)
    return CPT(unitcell, lattice, solver, pert, periodization)
end

end # module