using LinearAlgebra: axpby!, inv!, lu!, mul!
using QuantumLattices: volume
using TightBindingApproximation: velocity

"""
    Conductivity{B<:BrillouinZone}

The dc conductivity tensor task: the bubble approximation of the Kubo-Bastin formula with the permutation decomposition (Bonbien & Manchon, arXiv:2005.04678),

    σ = σˢᵘʳᶠ + σˢᵉᵃ
    σˢᵘʳᶠ_αα′ = (2π²e²/h)(1/SN)Σ_k ∫dω (-∂_ωf) Tr[v_α(k)A(k,ω)v_α′(k)A(k,ω)]       (symmetric, Fermi-surface)
    σˢᵉᵃ_αα′ = (e²/2h)(1/SN)Σ_k ∫dω f(ω) Tr[{v_α∂_ω(Gʳ+Gᵃ)v_α′ - (α↔α′)}(Gʳ-Gᵃ)]   (antisymmetric, Fermi-sea)

with S the unitcell volume, N the number of momenta, `A = (Gʳ-Gᵃ)/2πi` the spectral matrix, G the periodized quantum cluster Green's function, and `v_α(k) = ∂h₀(k)/∂k_α` the bare velocity vertex obtained from the `velocity` of `TBA(qct)`, the bare unitcell tight-binding system reconstructed from the quantum cluster theory frontend, so that the vertices are guaranteed to be consistent with the Green's function. Vertex corrections are neglected (exact in DMFT, an approximation for CPT/VCA). The returned tensors are in units of e²/h (per layer).

- `brillouinzone`: the momentum mesh over the **full Brillouin zone of the unitcell** (fixed-mesh sum replacing the BZ integral).
- `μ`: the chemical potential. Following the convention of [`Ω`](@ref) and [`expectation`](@ref), the Green's function is evaluated at `ω+μ` while the thermal weights and integration limits are expressed in the frequency relative to `μ`, so the Hamiltonian needs not contain the `-μN` term; leave `μ=0` when it already does. (Within the fixed quantum-number sector of the ED solvers this shift is exact even at finite temperature, since `-μN` is then a constant of the sector.)
- `β`: the inverse temperature; `Inf` for zero temperature (then `-∂f` becomes δ(ω) and `f` becomes θ(-ω)).
- `η`: the Lorentzian broadening of the Green's function (plays the role of a phenomenological scattering rate).
- `which`: which part(s) of the conductivity to compute, `:surf`, `:sea` or `:both`.

Run-time options (passed at the assignment call site, see `options`):
- `gauge` (default `:rcoordinate`): the gauge of the velocity vertices; the default is the gauge of  the periodized Green's function of [`QCT`](@ref).
- `rtol` (default `1e-6`), `atol` (default `0.0`): relative/absolute tolerance of the adaptive  quadrature (`quadgk`). The ω integrals are evaluated by `quadgk` over the infinite interval (`(-Inf, 0)` for the sea term at zero temperature, `(-Inf, Inf)` otherwise), with the momentum-mesh sum inside the integrand; the impurity solver caches its Green's function (and its inverse) at the latest frequency, so it is evaluated only once per quadrature node and shared across all momenta.
"""
struct Conductivity{B<:BrillouinZone}
    brillouinzone::B
    μ::Float64
    β::Float64
    η::Float64
    which::Symbol
    function Conductivity(brillouinzone::BrillouinZone; μ::Real=0.0, β::Real=Inf, η::Real=0.05, which::Symbol=:both)
        @assert which∈(:surf, :sea, :both) "Conductivity error: which must be `:surf`, `:sea` or `:both`."
        @assert η>0 "Conductivity error: η must be positive."
        @assert β>0 "Conductivity error: β must be positive."
        new{typeof(brillouinzone)}(brillouinzone, Float64(μ), Float64(β), Float64(η), which)
    end
end
@inline dependencytypes(::Type{<:Conductivity}) = ()
@inline options(::Type{<:Assignment{<:Conductivity}}) = (
    gauge = "gauge of the velocity vertices (`:rcoordinate` or `:icoordinate`)",
    rtol = "relative tolerance of the adaptive quadrature (quadgk)",
    atol = "absolute tolerance of the adaptive quadrature (quadgk)",
)

"""
    ConductivityData{B<:BrillouinZone} <: Data

The dc conductivity tensors (in units of e²/h per layer): `surf` (Fermi-surface/symmetric), `sea` (Fermi-sea/antisymmetric) and `total = surf + sea`, each a `dim×dim` matrix over the spatial directions.
"""
struct ConductivityData{B<:BrillouinZone} <: Data
    brillouinzone::B
    surf::Matrix{Float64}
    sea::Matrix{Float64}
    total::Matrix{Float64}
end
function run!(qct::Algorithm{<:QCT}, assignment::Assignment{<:Conductivity}; gauge::Symbol=:rcoordinate, rtol::Real=1e-6, atol::Real=0.0)
    @assert gauge∈(:rcoordinate, :icoordinate) "Conductivity error: gauge must be `:rcoordinate` or `:icoordinate`."
    frontend, task = qct.frontend, assignment.task
    bz = [k for k in task.brillouinzone]
    dim, Nk = length(first(bz)), length(bz)
    S = volume(frontend.unitcell)
    solver = frontend.solver
    # per-momentum precomputation
    Vs = [frontend.perturbation(k) for k in bz]
    H₀ = TBA(frontend)
    # velocity vertices mapped to the cluster space by the periodization, Wₐ(k) = U(k)'vₐ(k)U(k), so that `tr(vₐ(k)A(k)vₐ′(k)A(k)) = tr(Wₐ(k)Ã(k)Wₐ′(k)Ã(k))` with Ã=U'AU because UU'=I
    Ws = [[begin U = matrix(frontend.periodization, k); U'*velocity(H₀, k, α; gauge=gauge)*U end for k in bz] for α in 1:dim]
    # the conductivity tensors
    surf = zeros(Float64, dim, dim)
    sea = zeros(Float64, dim, dim)
    if isinf(task.β)
        # zero temperature: -∂f = δ(ω) collapses the surf term to ω=0
        if task.which∈(:surf, :both)
            n = size(first(Vs), 1)
            G̃, A = ntuple(i->Matrix{ComplexF64}(undef, n, n), 2)
            WAs = ntuple(i->Matrix{ComplexF64}(undef, n, n), dim)
            G⁻¹ = inv(solver, task.μ+1im*task.η)
            for ik in 1:Nk
                spectral!(A, periodized!(G̃, G⁻¹, Vs[ik]))
                for α in 1:dim
                    mul!(WAs[α], Ws[α][ik], A)
                end
                for α′ = 1:dim, α = 1:dim
                    # cluster-space equivalent of `surf[α, α′] += real(tr(vs[α][ik]*A*vs[α′][ik]*A))`
                    surf[α, α′] += real(trprod(WAs[α], WAs[α′]))
                end
            end
            surf .*= 2π^2/(S*Nk)
        end
        if task.which∈(:sea, :both)
            integrand = integrandˢᵉᵃ(solver, Ws, Vs, task.μ, task.β, task.η)
            vals = quadgk(integrand, -Inf, 0.0; rtol=rtol, atol=atol)[1]
            sea .= reshape(vals, dim, dim)
            sea .*= 1/(2S*Nk)
        end
    elseif task.which==:both
        integrand = integrandᵇᵒᵗʰ(solver, Ws, Vs, task.μ, task.β, task.η)
        vals = quadgk(integrand, -Inf, Inf; rtol=rtol, atol=atol)[1]
        surf .= reshape(vals[1:dim^2], dim, dim)
        sea .= reshape(vals[dim^2+1:end], dim, dim)
        surf .*= 2π^2/(S*Nk)
        sea .*= 1/(2S*Nk)
    elseif task.which==:surf
        integrand = integrandˢᵘʳᶠ(solver, Ws, Vs, task.μ, task.β, task.η)
        vals = quadgk(integrand, -Inf, Inf; rtol=rtol, atol=atol)[1]
        surf .= reshape(vals, dim, dim)
        surf .*= 2π^2/(S*Nk)
    else
        integrand = integrandˢᵉᵃ(solver, Ws, Vs, task.μ, task.β, task.η)
        vals = quadgk(integrand, -Inf, Inf; rtol=rtol, atol=atol)[1]
        sea .= reshape(vals, dim, dim)
        sea .*= 1/(2S*Nk)
    end
    return ConductivityData(task.brillouinzone, surf, sea, surf+sea)
end

# in-place evaluation of the cluster-space Green's function G̃ = inv(G⁻¹-V); the result aliases `G̃`
@inline function periodized!(G̃::Matrix{ComplexF64}, G⁻¹::AbstractMatrix{<:Number}, V::AbstractMatrix{<:Number})
    copyto!(G̃, G⁻¹)
    axpby!(-1, V, 1, G̃)
    return inv!(lu!(G̃))
end
# in-place cluster-space frequency derivative ∂G̃ = G̃ M G̃ with M = G⁻¹ ∂G G⁻¹
@inline function ∂periodized!(∂G̃::Matrix{ComplexF64}, G̃::AbstractMatrix{<:Number}, M::Matrix{ComplexF64}, t::Matrix{ComplexF64})
    mul!(t, M, G̃)
    mul!(∂G̃, G̃, t)
    return ∂G̃
end
# in-place spectral matrix A = (G̃-G̃')/(2πi)
@inline function spectral!(A::Matrix{ComplexF64}, G̃::AbstractMatrix{<:Number})
    n = size(G̃, 1)
    @inbounds for j in 1:n, i in 1:n
        A[i, j] = (G̃[i, j]-conj(G̃[j, i]))/(2π*1im)
    end
    return A
end
# in-place D = ∂G̃+∂G̃' and X = G̃-G̃'
@inline function DX!(D::Matrix{ComplexF64}, X::Matrix{ComplexF64}, G̃::AbstractMatrix{<:Number}, ∂G̃::AbstractMatrix{<:Number})
    n = size(G̃, 1)
    @inbounds for j in 1:n, i in 1:n
        D[i, j] = ∂G̃[i, j]+conj(∂G̃[j, i])
        X[i, j] = G̃[i, j]-conj(G̃[j, i])
    end
    return D, X
end

# the ω integrands for quadgk
function integrandˢᵘʳᶠ(solver, Ws, Vs, μ, β, η)
    dim, Nk = length(Ws), length(Vs)
    n = size(first(Vs), 1)
    G̃, A = ntuple(i->Matrix{ComplexF64}(undef, n, n), 2)
    WAs = ntuple(i->Matrix{ComplexF64}(undef, n, n), dim)
    return function (ω)
        z = ω + μ + 1im*η
        acc = zeros(ComplexF64, dim, dim)
        w = ∂fermi(β, ω)
        G⁻¹ = inv(solver, z)
        for ik in 1:Nk
            spectral!(A, periodized!(G̃, G⁻¹, Vs[ik]))
            for α in 1:dim
                mul!(WAs[α], Ws[α][ik], A)
            end
            for α′ = 1:dim, α = 1:dim
                # cluster-space equivalent of `acc[α, α′] += w*tr(vs[α][ik]*A*vs[α′][ik]*A)`
                acc[α, α′] += w*trprod(WAs[α], WAs[α′])
            end
        end
        return real.(vec(acc))
    end
end
function integrandˢᵉᵃ(solver, Ws, Vs, μ, β, η)
    dim, Nk = length(Ws), length(Vs)
    n = size(first(Vs), 1)
    G̃, ∂G̃, D, X, M, t = ntuple(i->Matrix{ComplexF64}(undef, n, n), 6)
    WDs = ntuple(i->Matrix{ComplexF64}(undef, n, n), dim)
    WXs = ntuple(i->Matrix{ComplexF64}(undef, n, n), dim)
    return function (ω)
        z = ω + μ + 1im*η
        acc = zeros(ComplexF64, dim, dim)
        w = isinf(β) ? 1.0 : fermi(β, ω)
        G⁻¹, ∂G = inv(solver, z), derivative(solver, z)
        # M = G⁻¹ ∂G G⁻¹ is momentum independent and computed once per frequency
        mul!(t, ∂G, G⁻¹)
        mul!(M, G⁻¹, t)
        for ik in 1:Nk
            G̃m = periodized!(G̃, G⁻¹, Vs[ik])
            ∂periodized!(∂G̃, G̃m, M, t)
            DX!(D, X, G̃m, ∂G̃)
            for α in 1:dim
                mul!(WDs[α], Ws[α][ik], D)
                mul!(WXs[α], Ws[α][ik], X)
            end
            for α′ = 1:dim, α = 1:dim
                # cluster-space equivalent of `acc[α, α′] += w*tr((vs[α][ik]*D*vs[α′][ik]-vs[α′][ik]*D*vs[α][ik])*X)`
                acc[α, α′] += w*(trprod(WDs[α], WXs[α′])-trprod(WDs[α′], WXs[α]))
            end
        end
        return real.(vec(acc))
    end
end
function integrandᵇᵒᵗʰ(solver, Ws, Vs, μ, β, η)
    dim, Nk = length(Ws), length(Vs)
    n = size(first(Vs), 1)
    G̃, ∂G̃, A, D, X, M, t = ntuple(i->Matrix{ComplexF64}(undef, n, n), 7)
    WAs = ntuple(i->Matrix{ComplexF64}(undef, n, n), dim)
    WDs = ntuple(i->Matrix{ComplexF64}(undef, n, n), dim)
    WXs = ntuple(i->Matrix{ComplexF64}(undef, n, n), dim)
    return function (ω)
        z = ω + μ + 1im*η
        accˢᵘʳᶠ = zeros(ComplexF64, dim, dim)
        accˢᵉᵃ = zeros(ComplexF64, dim, dim)
        ∂f, f = ∂fermi(β, ω), fermi(β, ω)
        G⁻¹, ∂G = inv(solver, z), derivative(solver, z)
        mul!(t, ∂G, G⁻¹)
        mul!(M, G⁻¹, t)
        for ik in 1:Nk
            G̃m = periodized!(G̃, G⁻¹, Vs[ik])
            ∂periodized!(∂G̃, G̃m, M, t)
            spectral!(A, G̃m)
            DX!(D, X, G̃m, ∂G̃)
            for α in 1:dim
                mul!(WAs[α], Ws[α][ik], A)
                mul!(WDs[α], Ws[α][ik], D)
                mul!(WXs[α], Ws[α][ik], X)
            end
            for α′ = 1:dim, α = 1:dim
                # cluster-space equivalents of
                # `accˢᵘʳᶠ[α, α′] += ∂f*tr(vs[α][ik]*A*vs[α′][ik]*A)`
                # `accˢᵉᵃ[α, α′] += f*tr((vs[α][ik]*D*vs[α′][ik]-vs[α′][ik]*D*vs[α][ik])*X)`
                accˢᵘʳᶠ[α, α′] += ∂f*trprod(WAs[α], WAs[α′])
                accˢᵉᵃ[α, α′] += f*(trprod(WDs[α], WXs[α′])-trprod(WDs[α′], WXs[α]))
            end
        end
        return vcat(real.(vec(accˢᵘʳᶠ)), real.(vec(accˢᵉᵃ)))
    end
end

# fermi function and its derivative, with the large-|x| cutoff to avoid overflow in exp(βω)
@inline fermi(β::Float64, ω::Real) = ω>=0 ? (x = exp(-β*ω); x/(1+x)) : inv(1+exp(β*ω))
@inline ∂fermi(β::Float64, ω::Real) = (x = β*ω/2; abs(x)>40 ? 0.0 : β/4*sech(x)^2)
