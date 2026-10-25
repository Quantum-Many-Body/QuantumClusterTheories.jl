module QuantumClusterTheoriesPlotsExt

using QuantumClusterTheories: DynamicalSpectra
using QuantumLattices: Assignment, BrillouinZone, ReciprocalZone, rank,str
using RecipesBase: @recipe

"""
    @recipe plot(assignment::Assignment{<:DynamicalSpectra})

Define the recipe for the visualization of the dynamical spectra of a quantum-cluster-theory system: multi-panel heatmaps along the reciprocal path for a task with multiple energies (one panel per set of generalized orbitals), or the spectral surface on a two-dimensional Brillouin/reciprocal zone for a single-energy task.
"""
@recipe function plot(assignment::Assignment{<:DynamicalSpectra})
    title --> str(assignment)
    titlefontsize --> 10
    data = assignment.data
    if data.energies isa Float64
        reciprocalspace = data.reciprocalspace
        isa(reciprocalspace, Union{BrillouinZone, ReciprocalZone}) && rank(reciprocalspace)==2 || error("plot error: the single-energy dynamical spectra can only be visualized on a two-dimensional Brillouin zone or reciprocal zone.")
        return reciprocalspace, reshape(data.values, length(range(reciprocalspace, 2)), length(range(reciprocalspace, 1)), size(data.values, 3))
    else
        return Tuple(data)
    end
end

end # module
