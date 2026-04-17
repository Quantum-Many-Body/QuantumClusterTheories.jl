"""
    const qcttimer

Default shared timer for all quantum cluster theory methods.
"""
const qcttimer = TimerOutput()

"""
    operators(tbakind::TBAKind{:TBA}, lattice::AbstractLattice, hilbert::Hilbert{<:Fock}, table::Table=Table(hilbert, Metric(tbakind, hilbert))) -> Vector{<:CoordinatedIndex}
    operators(tbakind::TBAKind{:BdG}, lattice::AbstractLattice, hilbert::Hilbert{<:Fock}, table::Table=Table(hilbert, Metric(tbakind, hilbert))) -> Vector{<:CoordinatedIndex}

Get the single-particle operators sorted by table index.
For TBA kind, returns only annihilation operators; for BdG kind, returns all operators.
"""
function operators(tbakind::TBAKind{:TBA}, lattice::AbstractLattice, hilbert::Hilbert{<:Fock}, table::Table=Table(hilbert, Metric(tbakind, hilbert)))
    result = [CoordinatedIndex(Index(site, fockindex), coordinate, zero(coordinate)) for (site, coordinate) in enumerate(lattice) for fockindex in hilbert[site] if isannihilation(fockindex)]
    return sort!(result; by=index->table[index])
end
function operators(tbakind::TBAKind{:BdG}, lattice::AbstractLattice, hilbert::Hilbert{<:Fock}, table::Table=Table(hilbert, Metric(tbakind, hilbert)))
    result = [CoordinatedIndex(Index(site, fockindex), coordinate, zero(coordinate)) for (site, coordinate) in enumerate(lattice) for fockindex in hilbert[site]]
    return sort!(result; by=index->table[index])
end

"""
    quadratic(terms::OneAtLeast{Term}) -> Tuple

Extract the quadratic (rank-2) terms from a collection of terms.
"""
@generated quadratic(terms::OneAtLeast{Term}) = Expr(:tuple, [:(terms[$i]) for (i, T) in enumerate(fieldtypes(terms)) if rank(T)==2]...)

"""
    ImpuritySolver

Abstract type for impurity solvers used in quantum cluster theory calculations.
Subtypes must implement the call syntax `solver(ω)` to return the solver's response function at frequency `ω`.
"""
abstract type ImpuritySolver end

"""
    inv(solver::ImpuritySolver, ω::Number) -> Matrix{ComplexF64}

Get the inverse of the retarded Green's function at frequency `ω`
"""
@inline Base.inv(solver::ImpuritySolver, ω::Number) = inv(solver(ω))

"""
    Perturbation

Abstract type for perturbations in quantum cluster theory.
Subtypes must implement `kind`, `update!`, and the call syntax `perturbation(k)` returning the perturbation matrix at momentum `k`.
"""
abstract type Perturbation end
@inline kind(perturbation::Perturbation) = kind(typeof(perturbation))

"""
    CPTPerturbation{V<:TBA} <: Perturbation

CPT (cluster perturbation theory) perturbation containing the intercluster quadratic terms of a system.
"""
struct CPTPerturbation{V<:TBA} <: Perturbation
    intercluster::V
end
@inline kind(::Type{<:CPTPerturbation{V}}) where {V<:TBA} = kind(V)
@inline update!(perturbation::CPTPerturbation; parameters...) = (update!(perturbation.intercluster; parameters...); perturbation)
@inline (perturbation::CPTPerturbation)(k::Union{AbstractVector{<:Number}, Nothing}=nothing) = matrix(perturbation.intercluster, k)

"""
    Perturbation(lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}; neighbors::Union{Int, Neighbors}=nneighbor(terms)) -> CPTPerturbation
    Perturbation(bonds::AbstractVector{<:Bond}, hilbert::Hilbert, terms::OneOrMore{Term}) -> CPTPerturbation

Construct a CPT perturbation from lattice or bonds, hilbert space and terms.
"""
@inline function Perturbation(lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}; neighbors::Union{Int, Neighbors}=nneighbor(terms))
    return Perturbation(filter!(!isintracell, bonds(lattice, neighbors)), hilbert, terms)
end
function Perturbation(bonds::AbstractVector{<:Bond}, hilbert::Hilbert, terms::OneOrMore{Term})
    terms = quadratic(OneOrMore(terms))
    kind = TBAKind(typeof(terms), valtype(hilbert))
    H = Generator(bonds, hilbert, terms, plain, lazy; half=false)
    quadraticization = Quadraticization{typeof(kind)}(Table(hilbert, Metric(kind, hilbert)))
    commt = commutator(kind, hilbert)
    return CPTPerturbation(TBA{typeof(kind)}(H, quadraticization, commt))
end

"""
    VCAPerturbation{V<:TBA, W<:TBA} <: Perturbation

VCA (Variational Cluster Approach) perturbation containing both intercluster quadratic terms and Weiss field terms of a system.
"""
struct VCAPerturbation{V<:TBA, W<:TBA} <: Perturbation
    intercluster::V
    weiss::W
end
@inline kind(::Type{<:VCAPerturbation{V}}) where {V<:TBA} = kind(V)
@inline function update!(perturbation::VCAPerturbation; parameters...)
    update!(perturbation.intercluster; parameters...)
    update!(perturbation.weiss; parameters...)
    return perturbation
end
@inline function (perturbation::VCAPerturbation)(k::Union{AbstractVector{<:Number}, Nothing}=nothing)
    return matrix(perturbation.intercluster, k) - matrix(perturbation.weiss, k)
end

"""
    Perturbation(
        lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, weiss::OneOrMore{Term};
        neighbors::Union{Int, Neighbors}=max(nneighbor(terms), nneighbor(weiss))
    ) -> VCAPerturbation
    Perturbation(
        bonds₁::AbstractVector{<:Bond}, bonds₂::AbstractVector{<:Bond}, hilbert::Hilbert, terms::OneOrMore{Term}, weiss::OneOrMore{Term}
    ) -> VCAPerturbation

Construct a VCA perturbation from lattice, hilbert space, terms, and Weiss field terms.
"""
@inline function Perturbation(
    lattice::AbstractLattice, hilbert::Hilbert, terms::OneOrMore{Term}, weiss::OneOrMore{Term};
    neighbors::Union{Int, Neighbors}=max(nneighbor(terms), nneighbor(weiss))
)
    bonds₀ = bonds(lattice, neighbors)
    bonds₁, bonds₂ = eltype(bonds₀)[], eltype(bonds₀)[]
    for bond in bonds₀
        isintracell(bond) ? push!(bonds₂, bond) : push!(bonds₁, bond)
    end
    return Perturbation(bonds₁, bonds₂, hilbert, terms, weiss)
end
function Perturbation(bonds₁::AbstractVector{<:Bond}, bonds₂::AbstractVector{<:Bond}, hilbert::Hilbert, terms::OneOrMore{Term}, weiss::OneOrMore{Term})
    terms = quadratic(OneOrMore(terms))
    weiss = quadratic(OneOrMore(weiss))
    kind = TBAKind(typeof((terms..., weiss...)), valtype(hilbert))
    H = Generator(bonds₁, hilbert, terms, plain, lazy; half=false)
    W = Generator(bonds₂, hilbert, weiss, plain, lazy; half=false)
    quadraticization = Quadraticization{typeof(kind)}(Table(hilbert, Metric(kind, hilbert)))
    commt = commutator(kind, hilbert)
    return VCAPerturbation(TBA{typeof(kind)}(H, quadraticization, commt), TBA{typeof(kind)}(W, quadraticization, commt))
end

"""
    Periodization{N}

Structure for crystallographic periodization in quantum cluster theory.
Stores the coordinates of lattice operators and groups them by equivalence under lattice translations.
"""
struct Periodization{N}
    coordinates::Vector{SVector{N, Float64}}
    groups::Vector{Vector{Int}}
end

"""
    Periodization(
        ops_lattice::AbstractVector{<:CoordinatedIndex}, ops_unitcell::AbstractVector{<:CoordinatedIndex}, vectors::AbstractVector{<:AbstractVector{<:Number}};
        atol=atol, rtol=rtol
    ) -> Periodization

Construct a `Periodization` object by grouping lattice operators into translation-equivalent sets.
"""
function Periodization(ops_lattice::AbstractVector{<:CoordinatedIndex}, ops_unitcell::AbstractVector{<:CoordinatedIndex}, vectors::AbstractVector{<:AbstractVector{<:Number}}; atol=atol, rtol=rtol)
    coordinates = map(rcoordinate, ops_lattice)
    groups = Vector{Int}[]
    for op_unitcell in ops_unitcell
        group = Int[]
        for (i, op_lattice) in enumerate(ops_lattice)
            op_unitcell.index.internal == op_lattice.index.internal && issubordinate(rcoordinate(op_unitcell)-rcoordinate(op_lattice), vectors; atol, rtol) && push!(group, i)
        end
        push!(groups, group)
    end
    return Periodization(coordinates, groups)
end

"""
    (periodization::Periodization)(data::AbstractMatrix{<:Number}, k::AbstractVector{<:Number}) -> Matrix{ComplexF64}

Apply crystallographic periodization to data at a given crystal momentum `k`.
"""
function (periodization::Periodization)(data::AbstractMatrix{<:Number}, k::AbstractVector{<:Number})
    result = zeros(ComplexF64, length(periodization.groups), length(periodization.groups))
    for (i, groupᵢ) in enumerate(periodization.groups), (j, groupⱼ) in enumerate(periodization.groups)
        for m in groupᵢ, n in groupⱼ
            result[i, j] += data[m, n] * exp(1im*dot(k, periodization.coordinates[n]-periodization.coordinates[m]))
        end
    end
    L = count(periodization)
    for index in eachindex(result)
        result[index] /= L
    end
    return result
end

"""
    count(periodization::Periodization) -> Int

Return the number of unit cells in the cluster (i.e., the ratio of cluster size to unitcell size).
"""
@inline Base.count(periodization::Periodization) = length(periodization.coordinates)÷length(periodization.groups)

"""
    QCT{U<:AbstractLattice, L<:AbstractLattice, I<:ImpuritySolver, V<:Perturbation, P<:Periodization} <: Frontend

Quantum cluster theory frontend that combines an impurity solver, perturbation, and crystallographic periodization to compute the Green's function of a system.
"""
struct QCT{U<:AbstractLattice, L<:AbstractLattice, I<:ImpuritySolver, V<:Perturbation, P<:Periodization} <: Frontend
    unitcell::U
    lattice::L
    solver::I
    perturbation::V
    periodization::P
end
@inline Parameters(qct::QCT) = Parameters(qct.solver)
@inline function update!(qct::QCT; parameters...)
    update!(qct.solver; parameters...)
    update!(qct.perturbation; parameters...)
    return qct
end

"""
    CPT{U<:AbstractLattice, L<:AbstractLattice, I<:ImpuritySolver, V<:CPTPerturbation, P<:Periodization}

Alias for [`QCT`](@ref) with `V<:CPTPerturbation`, representing cluster perturbation theory.
"""
const CPT{U<:AbstractLattice, L<:AbstractLattice, I<:ImpuritySolver, V<:CPTPerturbation, P<:Periodization} = QCT{U, L, I, V, P}

"""
    VCA{U<:AbstractLattice, L<:AbstractLattice, I<:ImpuritySolver, V<:VCAPerturbation, P<:Periodization}

Alias for [`QCT`](@ref) with `V<:VCAPerturbation`, representing the Variational Cluster Approach (VCA).
"""
const VCA{U<:AbstractLattice, L<:AbstractLattice, I<:ImpuritySolver, V<:VCAPerturbation, P<:Periodization} = QCT{U, L, I, V, P}

"""
    (qct::Union{QCT, Algorithm{<:QCT}})(ω::Number, k::Union{AbstractVector{<:Number}, Nothing}=nothing; periodization::Bool=true)

Evaluate the Green's function at frequency `ω` and momentum `k` by use of quantum cluster theory.
When `k` is `nothing`, no periodization is performed even if `periodization=true`.
"""
@inline (qct::Algorithm{<:QCT})(ω::Number, k::Union{AbstractVector{<:Number}, Nothing}=nothing; periodization::Bool=true) = (qct.frontend)(ω, k; periodization)
function (qct::QCT)(ω::Number, k::Union{AbstractVector{<:Number}, Nothing}=nothing; periodization::Bool=true)
    G⁻¹, V = inv(qct.solver, ω), qct.perturbation(k)
    result = inv(G⁻¹-V)
    !isnothing(k) && periodization && (result = qct.periodization(result, k))
    return result
end

"""
    Ω(qct::Algorithm{<:QCT}; kwargs...) -> Float64
    Ω(
        qct::QCT;
        brillouinzone::BrillouinZone=BrillouinZone(reciprocals(qct.lattice), 100), μ::Real=0.0, atol::Real=1e-6, rtol::Real=1e-6, maxevals::Int=10^6
    ) -> Float64

Compute the grand potential per unit cell of the quantum cluster theory system.

For `Algorithm{<:VCA}`, this delegates to the second method.
"""
@inline Ω(qct::Algorithm{<:QCT}; kwargs...) = Ω(qct.frontend; kwargs...)
function Ω(qct::QCT; brillouinzone::BrillouinZone=BrillouinZone(reciprocals(qct.lattice), 100), μ::Real=0.0, atol::Real=1e-6, rtol::Real=1e-6, maxevals::Int=10^6)
    Vs = [qct.perturbation(k) for k in brillouinzone]
    function f(ω::Real)
        result = 0.0
        G = qct.solver(1im*ω+μ)
        for V in Vs
            result -= log(abs(det(I-V*G)))
        end
        return result / length(brillouinzone) / π
    end
    part₁ = quadgk(f, 0, Inf; atol, rtol, maxevals)[1]
    part₂ = real(mapreduce(tr, +, Vs)) / length(brillouinzone) / 2
    result = (Ω(qct.solver) + part₁ + part₂) / count(qct.periodization)
    return result::Float64
end

"""
    expectation(qct::Algorithm{<:QCT}, m::Union{AbstractMatrix{<:Number}, Function, Symbol}; kwargs...) -> Float64
    expectation(
        qct::QCT, m::Union{AbstractMatrix{<:Number}, Function, Symbol};
        brillouinzone::BrillouinZone=BrillouinZone(reciprocals(qct.lattice), 100), μ::Real=0.0, p::Real=1.0, atol::Real=1e-6, rtol::Real=1e-6, maxevals::Int=10^6
    ) -> Float64

Compute the expectation value of an operator `m` (matrix, or function of `k`, or the symbol specifies the Weiss term in VCA) over the quantum cluster theory system by integrating over the Brillouin zone and frequency.

For `Algorithm{<:QCT}`, this delegates to the second method.
"""
@inline expectation(qct::Algorithm{<:QCT}, m::Union{AbstractMatrix{<:Number}, Function, Symbol}; kwargs...) = expectation(qct.frontend, m; kwargs...)
function expectation(qct::VCA, m::Symbol; kwargs...)
    weiss = update!(deepcopy(qct.perturbation.weiss); m=>1)
    selected = TBA{typeof(kind(weiss))}(expand(weiss.system, m), weiss.quadraticization, weiss.commutator)
    f(k::AbstractVector{<:Number}) = matrix(selected, k).H.data
    return expectation(qct, f; kwargs...)
end
function expectation(
    qct::QCT, m::Union{AbstractMatrix{<:Number}, Function};
    brillouinzone::BrillouinZone=BrillouinZone(reciprocals(qct.lattice), 100), μ::Real=0.0, p::Real=1.0, atol::Real=1e-6, rtol::Real=1e-6, maxevals::Int=10^6
)
    Vs = [qct.perturbation(k) for k in brillouinzone]
    Ss = [_matrix_(m, k) for k in brillouinzone]
    Ts = [tr(S) for S in Ss]
    function f(ω::Real)
        result = 0.0
        G⁻¹ = inv(qct.solver, 1im*ω+μ)
        for (V, S, T) in zip(Vs, Ss, Ts)
            result += real(tr(S*inv(G⁻¹-V)) - T/(1im*ω-p))
        end
        return result / length(brillouinzone) / π
    end
    result = quadgk(f, 0, Inf; atol, rtol, maxevals)[1] / length(qct.lattice)
    return result::Float64
end
@inline _matrix_(m::AbstractMatrix{<:Number}, ::AbstractVector{<:Number}) = m
@inline _matrix_(m::Function, k::AbstractVector{<:Number}) = m(k)::AbstractMatrix{<:Number}

"""
    optimize!(vca::Algorithm{<:VCA}; kwargs...)
    optimize!(
        vca::VCA;
        verbose=false, method=LBFGS(), options=Options(x_abstol=1e-4, x_reltol=1e-4, f_abstol=2e-6, f_reltol=2e-6),
        Ω_options=(brillouinzone=BrillouinZone(reciprocals(vca.lattice), 100), μ=0.0, atol=1e-6, rtol=1e-6, maxevals=10^6)
    )

Optimize the variational cluster approximation to find the stationary point of the grand potential.

For `Algorithm{<:VCA}`, this delegates to the second method.
For `VCA`, the parameters are optimized using the specified method.
"""
@inline optimize!(vca::Algorithm{<:VCA}; kwargs...) = optimize!(vca.frontend; kwargs...)
function optimize!(
    vca::VCA;
    verbose=false, method=LBFGS(), options=Options(x_abstol=1e-4, x_reltol=1e-4, f_abstol=2e-6, f_reltol=2e-6),
    Ω_options=(brillouinzone=BrillouinZone(reciprocals(vca.lattice), 100), μ=0.0, atol=1e-6, rtol=1e-6, maxevals=10^6)
    )
    params = Parameters(vca.perturbation.weiss)
    function f(v::Vector{<:Number})
        parameters = Parameters{keys(params)}(v...)
        update!(vca; parameters...)
        verbose && println(parameters)
        return Ω(vca; Ω_options...)
    end
    op = optimize(f, collect(map(real, values(params))), method, options)
    parameters = Parameters{keys(params)}(op.minimizer...)
    update!(vca; parameters...)
    return vca, op
end

"""
    DynamicalSpectra{R<:ReciprocalSpace} <: Action

Dynamical spectra.
"""
struct DynamicalSpectra{R<:ReciprocalSpace} <: Action
    reciprocalspace::R
    energies::Vector{Float64}
    DynamicalSpectra(reciprocalspace::ReciprocalSpace, energies::AbstractVector{<:Real}) = new{typeof(reciprocalspace)}(reciprocalspace, energies)
end
@inline options(::Type{<:Assignment{<:DynamicalSpectra}}) = (
    η = "Lorentz broadening",
    rescale = "function used to rescale the intensity of the spectrum at each energy-momentum point"
)

"""
    DynamicalSpectraData{R<:ReciprocalSpace} <: Data

Data of dynamical spectra computed from quantum cluster theory, including:

1) `reciprocalspace::R`: reciprocal space on which the spectra are computed.
2) `energies::Vector{Float64}`: energy sample points.
3) `values::Matrix{Float64}`: spectral function A(ω,k) = -Im[Tr[G(ω,k)]] at each energy-momentum point.
"""
struct DynamicalSpectraData{R<:ReciprocalSpace} <: Data
    reciprocalspace::R
    energies::Vector{Float64}
    values::Matrix{Float64}
end
function run!(qct::Algorithm{<:QCT}, ds::Assignment{<:DynamicalSpectra}; η::Real=0.1, rescale::Function=identity, options...)
    result = zeros(length(ds.action.energies), length(ds.action.reciprocalspace))
    Vs = [qct.frontend.perturbation(k) for k in ds.action.reciprocalspace]
    for (i, ω) in enumerate(ds.action.energies)
        G⁻¹ = inv(qct.frontend.solver, ω+1im*η)
        for (j, (k, V)) in enumerate(zip(ds.action.reciprocalspace, Vs))
            result[i, j] = rescale(-imag(tr(qct.frontend.periodization(inv(G⁻¹-V), k))))
        end
    end
    return DynamicalSpectraData(ds.action.reciprocalspace, ds.action.energies, result)
end
