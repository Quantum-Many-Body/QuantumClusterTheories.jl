module QuantumClusterTheoriesPlotsExt

using QuantumClusterTheories: DynamicalSpectra
using QuantumLattices: Assignment, BrillouinZone, ReciprocalZone, rank,str
using RecipesBase: @recipe

"""
    @recipe plot(assignment::Assignment{<:DynamicalSpectra})

Define the recipe for the visualization of the dynamical spectra of a quantum-cluster-theory system: heatmaps along the reciprocal path for a task with multiple energies, or the spectral surface on a two-dimensional Brillouin/reciprocal zone for a single-energy task. A single set of generalized orbitals is drawn as one heatmap (which can be overlaid onto an existing plot), while multiple sets are drawn as multi-panel heatmaps (one panel per set).
"""
@recipe function plot(assignment::Assignment{<:DynamicalSpectra})
    title --> str(assignment)
    titlefontsize --> 10
    data = assignment.data
    if data.energies isa Float64
        reciprocalspace = data.reciprocalspace
        isa(reciprocalspace, Union{BrillouinZone, ReciprocalZone}) && rank(reciprocalspace)==2 || error("plot error: the single-energy dynamical spectra can only be visualized on a two-dimensional Brillouin zone or reciprocal zone.")
        values = reshape(data.values, length(range(reciprocalspace, 2)), length(range(reciprocalspace, 1)), size(data.values, 3))
        return reciprocalspace, size(values, 3)==1 ? values[:, :, 1] : values
    elseif size(data.values, 3)==1
        return data.reciprocalspace, data.energies, data.values[:, :, 1]
    else
        return Tuple(data)
    end
end

end # module
