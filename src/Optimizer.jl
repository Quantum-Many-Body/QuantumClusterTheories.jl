module Optimizer

using LinearAlgebra: dot, norm
import Optim

export NoisyNewton

# Noisy Newton method in search of stationary points that is compatible with the interfaces defined by Optim.jl
# Method definition
struct NoisyNewton <: Optim.SecondOrderOptimizer
    step::Vector{Float64}
    accuracy::Vector{Float64}
end
@inline NoisyNewton(; step=Float64[], accuracy=Float64[]) = NoisyNewton(step, accuracy)

# State definition
mutable struct NoisyNewtonState{R<:Real} <: Optim.AbstractOptimizerState
    x::Vector{R}
    x_previous::Vector{R}
    f_x::R
    f_x_previous::R
    g_x::Vector{R}
    H_x::Matrix{R}
    const step::Vector{R}
    const step₀::Vector{R}
    const accuracy::Vector{R}
    iteration::Int
end

# Interface with Optim.jl
function Optim.optimize(f::Function, x₀::AbstractVector{<:Number}, method::NoisyNewton, options::Optim.Options)
    step = isempty(method.step) ? fill(1e-3, length(x₀)) : method.step
    f_x, g_x, H_x = value_gradient_hessian(f, x₀, step)
    d = Optim.TwiceDifferentiable(f, g_x, H_x, x₀, f_x; inplace=false)
    accuracy = isempty(method.accuracy) ? fill(1e-6, length(x₀)) : method.accuracy
    state = NoisyNewtonState(copy(x₀), copy(x₀), f_x, f_x, g_x, H_x, step, copy(step), accuracy, 0)
    return Optim.optimize(d, x₀, method, options, state)
end
function Optim.update_state!(d, state::NoisyNewtonState, method::NoisyNewton)
    if state.iteration > 0
        f_x, g_x, H_x = value_gradient_hessian(d.f, state.x, state.step)
        state.f_x_previous = state.f_x
        state.f_x = f_x
        state.g_x = g_x
        state.H_x = H_x
    end
    dx = -state.H_x \ state.g_x
    state.x_previous = copy(state.x)
    state.x += dx
    state.iteration += 1
    multiplier = 2.0
    for i in eachindex(state.step)
        state.step[i] = abs(dx[i])
        state.step[i] > state.step₀[i] && (state.step[i] = multiplier * state.step₀[i])
        state.step[i] < multiplier * state.accuracy[i] && (state.step[i] = multiplier * state.accuracy[i])
    end
    return false
end
function Optim.trace!(tr, d, state::NoisyNewtonState, iteration::Integer, ::NoisyNewton, options::Optim.Options, curr_time=time())
    dt = Dict()
    dt["time"] = curr_time
    if options.extended_trace
        dt["x"] = copy(state.x)
        dt["g(x)"] = copy(state.g_x)
        dt["h(x)"] = copy(state.H_x)
        dt["Current step"] = state.step
    end
    Optim.update!(tr, iteration, state.f_x, norm(state.g_x), dt, options.store_trace, options.show_trace, options.show_every)
end
@inline Optim.update_fgh!(d, state::NoisyNewtonState, method::NoisyNewton) = nothing

# Get the function value and the gradient as well as Hessian matrix by finite difference method.
function value_gradient_hessian(f::Function, x::AbstractVector{<:Number}, step::AbstractVector{<:Number})
    n = length(x)
    N = 1 + 2 * n + div(n * (n - 1), 2)
    y = zeros(Float64, N)
    xp = zeros(Float64, N, n)
    for k in 1:N
        xp[k, :] .= x
    end
    k = 1
    for i in 1:n
        k += 1
        xp[k, i] += step[i]
        k += 1
        xp[k, i] -= step[i]
    end
    for i in 1:n
        for j in 1:i-1
            k += 1
            xp[k, i] += 0.707 * (2 * ((i - 1) % 2) - 1) * step[i]
            xp[k, j] += 0.707 * (2 * ((j - 1) % 2) - 1) * step[j]
        end
    end
    y = [f(xp[i, :]) for i in 1:N]
    A = zeros(Float64, N, N)
    A[:, 2:n+1] = xp
    for k in 1:N
        A[k, 1] = 1.0
        m = n + 2
        for i in 1:n
            for j in 1:i
                A[k, m] = xp[k, i] * xp[k, j]
                if i == j
                    A[k, m] *= 0.5
                end
                m += 1
            end
        end
    end
    y = A \ y
    val = y[1]
    grad = y[2:n+1]
    val += dot(grad, x)
    hess = zeros(Float64, n, n)
    m = n + 2
    for i in 1:n
        for j in 1:i
            hess[i, j] = y[m]
            hess[j, i] = y[m]
            if i == j
                val += x[i] * x[j] * y[m] * 0.5
            else
                val += x[i] * x[j] * y[m]
            end
            m += 1
        end
    end
    grad += hess * x
    return val, grad, hess
end

end
