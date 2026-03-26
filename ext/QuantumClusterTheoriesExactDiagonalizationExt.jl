module QuantumClusterTheoriesExactDiagonalizationExt

using ExactDiagonalization: ED, Algorithm, GreenFunction, RetardedGreenFunction
using ExactDiagonalization: QuantumOperator, ExactMethod, prepare!
using QuantumClusterTheories: CPT, ImpuritySolver

"""
    EDSolver{O<:Number, G<:Union{GreenFunction{O}, RetardedGreenFunction{O}}} <: ImpuritySolver

An impurity solver based on exact diagonalization that wraps an ED algorithm and a GreenFunction.
"""
struct EDSolver{O<:Number, G<:Union{GreenFunction{O}, RetardedGreenFunction{O}}} <: ImpuritySolver
    ed::Algorithm{<:ED}
    greenfunction::G
end
@inline (solver::EDSolver)(ω::Number) = solver.greenfunction(ω)

"""
    EDSolver(ed::Algorithm{<:ED}, operators::AbstractVector{<:QuantumOperator}, method=ExactMethod(); kwargs...)

Construct an `EDSolver` from an ED algorithm and a list of operators.
"""
function EDSolver(ed::Algorithm{<:ED}, operators::AbstractVector{<:QuantumOperator}, method=ExactMethod(); kwargs...)
    prepare!(ed)
    greenfunction = RetardedGreenFunction(operators, ed, method; kwargs...)
    return EDSolver(ed, greenfunction)
end

end # module