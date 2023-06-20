#
# solvers.jl --
#
# Defines methods for solving inverse problems.
#
#------------------------------------------------------------------------------
#
# This file is part of InverseProblem



"""
    test_tol(x, x_last, atol, rtol)

yields a tolerance test between two values:
                    |x - x_tol| <= max(atol, rtol*|x_last|)

possible to give a `Tuple` `tol` instead of `atol` and `rtol`.

# Example
```julia
julia> test_tol(x, x_last, atol, rtol)
julia> tol = (atol, rtol)
julia> test_tol(x, x_last, tol)
```

"""
test_tol(x::X, x_last::X, atol::T, rtol::T) where {X<:Real,T<:Real} = 
         abs(x - x_last) <= max(atol, rtol*abs(x_last))
test_tol(x::X, x_last::X, atol::T, rtol::T) where {X<:AbstractArray,T<:Real} = 
         vnorm2(x - x_last) <= max(atol, rtol*vnorm2(x_last))
test_tol(x, x_last, tol::Tuple{Real,Real}) = test_tol(x, x_last, tol[1], tol[2])



"""
    QuasiNewton()

Structure of supertype `Solver` wich allows to apply the `solve` and `solve!` functions on
quasi-Newton problems. The optimizer VMLM-B from `OptimPackNextGen` is used to optimize the
loss function build with the `Cost` `S`.

An instance `quasinewton` of `QuasiNewton` is already build.

# Example
```julia
julia> solve!(x0, S, quasinewton [,cache=[]]; kwds...) # in place optimization
julia> solve(x0, S, quasinewton [,cache=[]]; kwds...) 
```

# Keywords
 - non_negative : (`false` by default) precise if the lower born of `vmlmb` must be at `0`.

"""
struct QuasiNewton <: Solver end

function solve!(x0::AbstractArray{T,N},
    S::Cost,
    ::QuasiNewton,
    cache::AbstractVector{T} = T[];
    keep_loss::Bool = false,
    non_negative::Bool = false,
    kwds...) where {T,N}

    function fg_solve!(x, g)
        loss = S(x, g)
        keep_loss && push!(cache, min(length(cache) == 0 ? Inf : cache[end], loss))
        return loss
    end

    if non_negative
        vmlmb!(fg_solve!, x0; lower=T(0), kwds...)
    else
        vmlmb!(fg_solve!, x0; kwds...)
    end
    
    return x0
end

const quasinewton = QuasiNewton()




"""
    BrentMin()

Structure of supertype `Solver` wich allows to apply the `solve` and `solve!` functions on
problems using the Brent method. The optimizer VMLM-B from `OptimPackNextGen` is used to optimize 
the loss function build with the `Cost` `S`.

An instance `brent` of `Brent` is already build.

# Example
```julia
julia> solve!(x0, S, brent [,cache=[]]; kwds...) # in place optimization
julia> solve(x0, S, brent [,cache=[]]; kwds...) 
```

# Keywords

"""
struct BrentMin <: Solver end

function solve!(x0::AbstractVector{T},
    S::Cost,
    ::BrentMin,
    cache::AbstractVector{T} = T[];
    keep_loss::Bool = false,
    estim_tol::Tuple{T,T} = (T(0.0), T(1e-3)),
    estim_bnds::AbstractVector{T} = [x0[1] - (x0[1]./2), x0[1] + (x0[1]./2)]) where {T}
    
    function f_solve(x)
        loss = S([x])
        keep_loss && push!(cache, min(length(cache) == 0 ? Inf : cache[end], loss))
        return loss
    end
    
    x0 = Brent.fmin(f_solve, estim_bnds[1][1], estim_bnds[2][1]; 
                    atol=estim_tol[1], rtol=estim_tol[2])[1]
    
    return x0
end

const brentmin = BrentMin()




"""
    PowellNewuoa()

Structure of supertype `Solver` wich allows to apply the `solve` and `solve!` functions on
problems by using the Newuoa method of Powell in `OptimPackNextGen` on a loss function
computed with the `Cost` `S` with `x0` as initialization.

An instance `powellnewuoa` of `PowellNewuoa` is already build.

# Example
```julia
julia> solve!(x0, S, powellnewuoa [,cache=[]]; kwds...) # in place optimization
julia> solve(x0, S, powellnewuoa [,cache=[]]; kwds...) 
```

# Keywords
 - trust_region : (`(1, 1e-3)` by default) gives the starting and ending size of the 
   trust region used by `Newuoa`.
 - scale : (`ones(x0)` by default) gives the scaling of the parameters of interest.

"""
struct PowellNewuoa <: Solver end

function solve!(x0::AbstractVector{T},
    S::Cost,
    ::PowellNewuoa,
    cache::AbstractVector{T} = T[];
    keep_loss::Bool = false,
    trust_region::Tuple{T,T} = (T(1.0), T(1e-3)),
    scale::AbstractVector{T} = ones(size(x0))) where {T}

    function f_solve(x)
        loss = S(x)
        keep_loss && push!(cache, min(length(cache) == 0 ? Inf : cache[end], loss))
        return loss
    end

    Newuoa.minimize!(f_solve, x0, trust_region[1], trust_region[2]; scale=scale)[2]
    
    return x0
end

const powellnewuoa = PowellNewuoa()




"""
    PowellBobyqa()

Structure of supertype `Solver` wich allows to apply the `solve` and `solve!` functions on
problems by using the Bobyqa method of Powell in `OptimPackNextGen` on a loss function
computed with the `Cost` `S` with `x0` as initialization.

An instance `powellbobyqa` of `PowellBobyqa` is already build.

# Example
```julia
julia> solve!(x0, S, powellbobyqa [,cache=[]]; kwds...) # in place optimization
julia> solve(x0, S, powellbobyqa [,cache=[]]; kwds...) 
```

# Keywords
 - trust_region : (`(1, 1e-3)` by default) gives the starting and ending size of the 
   trust region used by `Newuoa`.
 - estim_bnds : (`(-10, 10)` by default) gives the boundaries of the optimization 
   process.
 - scale : (`ones(x0)` by default) gives the scaling of the parameters of interest.

"""
struct PowellBobyqa <: Solver end

function solve!(x0::AbstractVector{T},
    S::Cost,
    ::PowellBobyqa,
    cache::AbstractVector{T} = T[];
    keep_loss::Bool = false,
    trust_region::Tuple{T,T} = (T(1.0), T(1e-3)),
    estim_bnds::AbstractVector{<:AbstractVector{T}} = [x0 .- (x0./2), x0 .+ (x0./2)],
    scale::AbstractVector{T} = ones(size(x0))) where {T}

    function f_solve(x)
        loss = S(x)
        keep_loss && push!(cache, min(length(cache) == 0 ? Inf : cache[end], loss))
        return loss
    end

    Bobyqa.minimize!(f_solve, x0, estim_bnds[1], estim_bnds[2], trust_region[1], trust_region[2]; 
                     scale=scale)
    
    return x0
end

const powellbobyqa = PowellBobyqa()




# """
#     alternated_solve!(x, y, Rx, Ry, form_Spbm_x, method_solver_x [,form_Spbm_y] 
#                       [,method_solver_y]; kwds...)

# gives the results (in-place) of an alternated estimation strategy of `x` and `y`, with
# `Rx` and `Ry` are respectively the regularization of `x` and `y`. 

# `form_Spbm_x` and `form_Spbm_y` must be `Function`s whiche yields a `SubProblem` structure
# containing the ingredients to estimate the unknowns. If `form_Spbm_y` is not specified, 
# `form_Spbm_x` is taken by default for `y`.

# `method_solver_x` and `method_solver_y` specify the method used to estimate the unknowns.
# If `method_solver_y` is not given, `method_solver_x` is taken by default for `y`.

# # Keywords
#  - maxiter : (`500` by default) specifies the maximum number of iteration of the alternated
#    strategy. 
#  - estim_tol : (`(0.0,1e-8)` by default) gives the absolute and relative tolerance which
#    define the stop criterion of the alternated strategy.
#  - auto_scale : (`false` by default) specifies if the tuning of the hyper-parameter must 
#    be done automatically via the scaling indetermination. To use this option, `Rx` and `Ry`
#    must be `HomogenRegul` structures.
#  - alpha : (`1.0` by default) is the initial scaling of the problem.
#  - alpha_tol : (`(0.0,1e-3` by default) gives the absolute and relative tolerance which 
#    define the stop criterion of the "warming step" of the auto-scale method.
#  - study : (`false` by default) if `true`, the method will also return the number of calls
#    of the direct model and the value of the loss function at each iteration, 
#    used for the optimization.
#  - cache : (`T[]` by default) if an outside `AbstractVector` is given, the method will store
#    every value of the loss function computed throughout the iterations.
#  - kwds_x : (`()` by default) is here to specify the keywords to give to the method 
#    used to estimate `x`.
#  - kwds_y : (`kwds_x` by default) is here to specify the keywords to give to the method 
#    used to estimate `y`.

# """
# function alternated_solve!(x::AbstractArray{T,N},
#     y::AbstractArray{T,N},
#     Rx::Regularization,
#     Ry::Regularization,
#     form_Spbm_x::Function,
#     method_solver_x::Solver,
#     form_Spbm_y::Function = form_Spbm_x,
#     method_solver_y::Solver = method_solver_x;
#     nb_max_iter::Int = 500,
#     estim_tol::Tuple{Real,Real} = (0.0, 1e-8),
#     auto_scale::Bool = false,
#     alpha::Real = 1.0,
#     alpha_tol::Tuple{Real,Real} = (0.0, 1e-3),
#     study::Val = Val(false),
#     cache::AbstractVector{T} = T[],
#     kwds_x::K = (),
#     kwds_y::K = kwds_x) where {T,N,K<:NamedTuple}
    
#     (auto_scale && typeof(Rx) == HomogenRegul{Real} && 
#      typeof(Ry) == HomogenRegul{Real}) && error("Regularizations must be homogeneous to have auto-scale.")

#     if study === Val(true)
#         nb_call = [1]
#         S_x_0 = form_Spbm_x(y, alpha^degree(Rx)*Rx)
#         S_y_0 = form_Spbm_y(x, (1/alpha^degree(Ry))*Ry)
#         losses = [S_x_0(x) + S_y_0.R(y)]
#     end
#     x_last = vcopy(x)
#     y_last = vcopy(y)
#     alpha_last = 0.0
#     loss_last = 0.0
#     iter = 0
#     while true
#         while true
#             # Update x
#             S_x = form_Spbm_x(y, alpha^degree(Rx)*Rx)
#             solve!(x, S_x, method_solver_x, cache; keep_loss=true, kwds_x...)
#             if auto_scale 
#                 # Update scaling
#                 alpha = best_scaling_factor(x, Rx, y, Ry)
#                 if iter > 0 || test_tol(alpha, alpha_last, alpha_tol)
#                     break
#                 end
#                 alpha_last = alpha
#             else
#                 break
#             end
#         end
#         # Update y
#         S_y = form_Spbm_y(x, (1/alpha^degree(Ry))*Ry)
#         solve!(y, S_y, method_solver_y, cache; keep_loss=true, kwds_y...)
#         # Update scaling
#         auto_scale && (alpha = best_scaling_factor(x, Rx, y, Ry))
#         if study === Val(true)
#             push!(nb_call, length(cache))
#             S_x_kp1 = form_Spbm_x(y, alpha^degree(Rx)*Rx)
#             push!(losses, S_y(y) + S_x_kp1.R(x))
#         end
#         # println(cache[end])
#         if iter >= nb_max_iter || test_tol(cache[end], loss_last, estim_tol) #|| 
#                                 #   (test_tol(x, x_last, estim_tol) && 
#                                 #    test_tol(y, y_last, estim_tol))
#             break
#         end
#         copyto!(x_last, x)
#         copyto!(y_last, y)
#         loss_last = cache[end]
#         iter += 1
#     end
#     # Post scaling
#     if auto_scale
#         vscale!(x, alpha)
#         vscale!(y, 1/alpha)
#     end
    
#     if study === Val(true)
#         return x, y, nb_call, losses
#     else
#         return x, y
#     end
# end




# """
#     best_scaling_factor(x, Rx, y, Ry)

# yields the optimal scaling factor α of the couple of solution `(α*x,y/α)`, according
# to the MAP problem, with `Rx` and `Ry` instances of `HomogenRegul`.

# """
# function best_scaling_factor(x::AbstractArray{T,N},
#     Rx::HomogenRegul,
#     y::AbstractArray{T,N},
#     Ry::HomogenRegul) where {T,N}
    
#     q_x = degree(Rx)
#     q_y = degree(Ry)

#     return ((q_y*call(Ry, y))/(q_x*call(Rx, x)))^(1/(q_x + q_y))
# end




"""
    NewtonRhapson()

Structure of supertype `Solver` wich allows to apply the `solve` and `solve!` functions on
problems using the Newton-Rhapson method. This method finds the fixed point of a function which
can be defined as a `Cost` `S`.

# Example
We look for the argument `x` of a function `f` which gives `f(x) = 0`. It is possible to define 
a `Cost` function `S` which will compute the value of a function 
`g(x) = x - f(x) + y`. Newton-Rhapson will find `x` which solves `g(x) = x` (corresponding to 
`f(x) = y`).
FIXME: update doc
"""
function newton_raphson_solve!(x::AbstractArray,
    g_x::AbstractArray,
    fg!::Function;
    nb_max_iter::Int = 500,
    estim_tol::Tuple{Real,Real} = (0.0, 1e-8),
    study::Val = Val(false))

    x_last = vcopy(x)
    if study === Val(true)
        estim_tol_evolution = Float64[]
    end
    iter = 0
    while true
        f_x = fg!(x, g_x)
        copyto!(x, x - g_x \ f_x)
        if study === Val(true)
            push!(estim_tol_evolution, vnorm2(x - x_last))
        end
        if iter >= nb_max_iter || test_tol(x, x_last, estim_tol)
            break
        end
        copyto!(x_last, x)
        iter += 1
    end

    if study === Val(true)
        return x, estim_tol_evolution
    else
        return x
    end
end


