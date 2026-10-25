module QuantumClusterTheoriesMakieExt

using QuantumClusterTheories: DynamicalSpectra
using QuantumLattices: Assignment, BrillouinZone, ReciprocalZone, rank, str
import Makie

"""
    Makie.plot(assignment::Assignment{<:DynamicalSpectra}; kwargs...)
    Makie.plot!(fig::Makie.Figure, assignment::Assignment{<:DynamicalSpectra}; kwargs...)

Visualize the dynamical spectra of a quantum-cluster-theory system: multi-panel heatmaps along the reciprocal path for a task with multiple energies (one panel per set of generalized orbitals), or the spectral surface on a two-dimensional Brillouin/reciprocal zone for a single-energy task.
"""
@inline Makie.plot(assignment::Assignment{<:DynamicalSpectra}; kwargs...) = Makie.plot!(Makie.Figure(), assignment; kwargs...)
function Makie.plot!(fig::Makie.Figure, assignment::Assignment{<:DynamicalSpectra}; kwargs...)
    data = assignment.data
    if data.energies isa Float64
        reciprocalspace = data.reciprocalspace
        isa(reciprocalspace, Union{BrillouinZone, ReciprocalZone}) && rank(reciprocalspace)==2 || error("plot error: the single-energy dynamical spectra can only be visualized on a two-dimensional Brillouin zone or reciprocal zone.")
        return Makie.plot!(fig, reciprocalspace, reshape(data.values, length(range(reciprocalspace, 2)), length(range(reciprocalspace, 1)), size(data.values, 3)); title=str(assignment), kwargs...)
    else
        return Makie.plot!(fig, data.reciprocalspace, data.energies, data.values; title=str(assignment), kwargs...)
    end
end

end # module
