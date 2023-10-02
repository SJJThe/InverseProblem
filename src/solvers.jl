#
# solvers.jl
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

The function can take scalars or arrays as input. 

# Example
```julia
julia> test_tol(x, x_last, atol, rtol)
julia> tol = (atol, rtol)
julia> test_tol(x, x_last, tol)
```
# Keywords
 - order : (`1` by default) defines the order of the norm used to test the
   tolerance when using `AbstractArray`.

"""
test_tol(x, x_last, tol::Tuple{Real,Real}; kwds...) = 
         test_tol(x, x_last, tol[1], tol[2]; kwds...)

test_tol(x::X, x_last::X, atol::T, rtol::T) where {X<:Real,T<:Real} = 
         abs(x - x_last) <= max(atol, rtol*abs(x_last))

function test_tol(x::X,
    x_last::X,
    atol::T,
    rtol::T;
    order::Int = 1) where {X<:AbstractArray,T<:Real}

    res = T(0)
    nrm = T(0)
    for i in eachindex(x, x_last)
        res += abs(x[i] - x_last[i])^order
        nrm[i] += abs(x_last[i])^order
    end

    return res^(1/order) <= max(atol, rtol*nrm^(1/order))
end




"""
    QuasiNewton()

Structure of supertype `Solver` wich allows to apply the `solve` and `solve!`
functions on quasi-Newton problems. The optimizer VMLM-B from `OptimPackNextGen`
is used to optimize the loss function build with the `Cost` `S`.

An instance `quasinewton` of `QuasiNewton` is already build.

# Example
```julia
julia> solve!(x0, S, quasinewton [,cache=[]]; kwds...) # in place optimization
julia> solve(x0, S, quasinewton [,cache=[]]; kwds...) 
```

# Keywords
 - `keep_loss` : (`false` by default) stores the result of `S(x)` in `cache`
   when true.
 - `non_negative` : (`false` by default) precise if the lower born of `vmlmb`
   must be at `0`.

See also: [`solve!`](@ref), [`Cost`](@ref), [`OptimPackNextGen`](@ref)

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

Structure of supertype `Solver` wich allows to apply the `solve` and `solve!`
functions on problems using the Brent fmin method defined in `OptimPackNextGen`
to optimize the cost function defined in `S`.

An instance `brent` of `Brent` is already build.

# Example
```julia
julia> solve!(x0, S, brent [,cache=[]]; kwds...) # in place optimization
julia> solve(x0, S, brent [,cache=[]]; kwds...) 
```

# Keywords
 - `keep_loss` : (`false` by default) stores the result of `S(x)` in `cache`
   when true.
 - `estim_tol` : (`(0,1e-3)` by default) defines the absolute and relative
   tolerance which are used to stop the optimization method.
 - `estim_bnds` : (x0 +/- (x0/2) by default) sets the boundaries of the method.

See also: [`solve!`](@ref), [`Cost`](@ref), [`OptimPackNextGen`](@ref)

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

Structure of supertype `Solver` wich allows to apply the `solve` and `solve!`
functions on problems by using the Newuoa method of Powell as defined in
`OptimPackNextGen` on a loss function computed with the `Cost` `S` with `x0` as
initialization.

An instance `powellnewuoa` of `PowellNewuoa` is already build.

# Example
```julia
julia> solve!(x0, S, powellnewuoa [,cache=[]]; kwds...) # in place optimization
julia> solve(x0, S, powellnewuoa [,cache=[]]; kwds...) 
```

# Keywords
 - `keep_loss` : (`false` by default) stores the result of `S(x)` in `cache`
   when true.
 - trust_region : (`(1, 1e-3)` by default) gives the starting and ending size of
   the trust region used by `Newuoa`.
 - scale : (`ones(x0)` by default) gives the scaling of the parameters of
   interest.

See also: [`solve!`](@ref), [`Cost`](@ref), [`OptimPackNextGen`](@ref)

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

Structure of supertype `Solver` wich allows to apply the `solve` and `solve!`
functions on problems by using the Bobyqa method of Powell as defined in
`OptimPackNextGen` on a loss function computed with the `Cost` `S` with `x0` as
initialization.

An instance `powellbobyqa` of `PowellBobyqa` is already build.

# Example
```julia
julia> solve!(x0, S, powellbobyqa [,cache=[]]; kwds...) # in place optimization
julia> solve(x0, S, powellbobyqa [,cache=[]]; kwds...) 
```

# Keywords
 - `keep_loss` : (`false` by default) stores the result of `S(x)` in `cache`
   when true.
 - trust_region : (`(1, 1e-3)` by default) gives the starting and ending size of the 
   trust region used by `Newuoa`.
 - estim_bnds : (`(-10, 10)` by default) gives the boundaries of the optimization 
   process.
 - scale : (`ones(x0)` by default) gives the scaling of the parameters of interest.

See also: [`solve!`](@ref), [`Cost`](@ref), [`OptimPackNextGen`](@ref)

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

    Bobyqa.minimize!(f_solve, x0, estim_bnds[1], estim_bnds[2], trust_region[1], 
                     trust_region[2]; scale=scale)
    
    return x0
end

const powellbobyqa = PowellBobyqa()




