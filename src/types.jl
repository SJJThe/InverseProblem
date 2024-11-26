#
# types.jl
#
# Defines generic structures common to inverse problems.
#
#------------------------------------------------------------------------------
#
# This file is part of InverseProblem



"""
    Cost

Abstract type which behaves like a `Mapping` type of `LazyAlgebra` and which 
refers to `call` and `call!` functions when applied to an element as an 
operator.

See also [`call`](@ref), [`call!`](@ref), [`LazyAlgebra.Mapping`](@ref),
[`Lkl`](@ref), [`InvProblem`](@ref), [`Regul`](@ref)

"""
abstract type Cost <: Mapping end

(C::Cost)(a, x) = call(a, C, x)
(C::Cost)(x) = call(1.0, C, x)
(C::Cost)(a, x::D, g::D; kwds...) where {D} = call!(a, C, x, g; kwds...)
(C::Cost)(x::D, g::D; kwds...) where {D} = call!(1.0, C, x, g; kwds...)



"""
    call!([a=1,] f, x, g [; incr=false])

gives back the value of a*f(x) while updating (in place) its gradient in `g`. 
`incr` is a boolean indicating if the gradient needs to be incremanted or 
reseted.

# Keywords
 - `incr`::Bool

"""
call!(f, x, g; kwds...) = call!(1.0, f, x, g; kwds...)

"""
    call([a::Real=1,] f, x)

yields the value of a*f(x).

"""
call(f, x) = call(1.0, f, x)




"""
    Lkl(A, d [, w=ones(size(d))])

yields a structure containing the elements essential to compute the Maximum likelihood 
criterion of an `AbstractArray` `x`, in the Gaussian hypothesis, that is:
                            (A(x) - d)'.Diag(w).(A(x) - d)
with Diag(w) the diagonal natrix whose diagonal elements are the one sotred in `w`.

Getters of an instance `L` can be imported with `InversePbm.` before:
 - model(L)       # gets the model `A`.
 - data(L)        # gets the data `d`.
 - weights(L)     # gets the weights `w` associated to the data `d`.
 - input_size(L)  # gets the input size that should correspond to the input size
   of `A` and to the size of `x`.
 - output_size(L) # gets the output size, that is the same size as `d`.

# Examples
 To apply it to an `AbstractArray` `x`, use:
```julia
julia> L([a,] x)             # apply call([a,] L, x)
julia> L([a,] x, g [; incr]) # apply call!([a,] L, x, g [; incr])
```

"""
struct Lkl{M<:Mapping,D<:AbstractArray} <: Cost
    A::M # model
    d::D # data
    w::D # weights of data

    function Lkl(A::M, d::D, w::D) where {M<:Mapping,D<:AbstractArray}
        @assert size(d) == size(w)
        return new{M,D}(A, d, w)
    end
end

Lkl(A, d::D) where{D<:AbstractArray} = Lkl(A, d, ones(size(d)))

model(L::Lkl) = L.A
data(L::Lkl) = L.d
weights(L::Lkl) = L.w
input_size(L::Lkl) = input_size(model(L))
output_size(L::Lkl) = size(data(L))


function call!(a::Real,
    L::Lkl,
    x::AbstractArray{T,N},
    g::AbstractArray{T,N};
    incr::Bool = false) where {T,N}
    
    A, d, w = model(L), data(L), weights(L)

    res = A(x) - d
    wres = w .*res
    lkl = a*vdot(res, wres)
    # do also g = (incr=false->0 : incr=true->1)*g + 2*A'*wres
    apply!(2*a, LazyAlgebra.Adjoint, A, wres, true, (incr ? 1 : 0), g)

    return Float64(lkl)
end

function call(a::Real,
    L::Lkl,
    x::AbstractArray{T,N}) where {T,N}
    
    A, d, w = model(L), data(L), weights(L)
    res = A(x) - d
    lkl = a*vdot(res, w, res)

    return Float64(lkl)
end




"""
    Regularization

Abstract type which obeys the `call!` and `call` laws, and behaves as a `Cost` type. 

"""
abstract type Regularization <: Cost end



"""
    Regul([mu=1,] f [, direct_inversion=false])

yields a structure of supertype `Regularization` containing the elements necessary 
to compute a regularization term `f` with it's multiplier `mu`. Creating an instance 
of Regul permits to use the `call` and `call!` functions that need to be
specialized. 

The `direct_inversion` boolean indicates if the regularization can
be directly inverted, in the context of the Normal equations. In that case, the
method `get_grad_op` can be called to yield the operator of the gradient that can
be applied directly.

Getters of an instance `R` can be imported with `InversePbm.` before:
 - multiplier(R)           # gets the multiplier `mu`.
 - func(R)                 # gets the computing process `f`.
 - use_direct_inversion(R) # indicates if it is possible to use the direct
   inversion method to solve the Normal equations.

# Example
```julia
julia> R = Regul(a, MyRegul)
Regul:
 - level `mu` : a
 - function `func` : MyRegul
 - use direct inversion `direct_inversion` : false
```
with `MyRegul` a strucure which defines the behaviors of `call([a,] MyRegul, x)` and 
`call!([a,] MyRegul, x, g [;incr])` on an `AbstractArray` `x` and its gradient `g`.

To apply it to `x`, use:
```julia
julia> R([a,] x)            # apply call([a,] R, x)
julia> R([a,] x, g [; incr]) # apply call!([a,] R, x, g [;incr])
```

Applying a scalar `b` to an instance `R` yields a new instance of `Regul`:
```julia
julia> R = b*Regul(a, MyRegul, true)
Regul:
 - level `mu` : b*a
 - function `func` : MyRegul
 - use direct inversion `direct_inversion` : true
```

Getting the gradient operator of an instance `R`, that can be used direclty in a
direct inversion scheme is done by calling:
```julia
julia> get_grad_op([a=1,] R)
```

"""
struct Regul{T<:Real,F} <: Regularization
    mu::T # multiplier
    f::F # function
    direct_inversion::Bool
end
multiplier(R::Regul) = R.mu
func(R::Regul) = R.f
use_direct_inversion(R::Regul) = R.direct_inversion

Regul(f, i::Bool) = Regul(1.0, f, i)
Regul(f) = Regul(1.0, f, false)

*(a::Real, R::Regul) = Regul(a*multiplier(R), func(R), use_direct_inversion(R))

Base.show(io::IO, R::Regul) = begin
    print(io,"Regul:")
    print(io,"\n - level `mu` : ",multiplier(R))
    print(io,"\n - function `func` : ",func(R))
    print(io,"\n - use direct inversion `direct_inversion` : ",use_direct_inversion(R))
end


function call!(a::Real,
    R::Regul,
    x::AbstractArray{T,N},
    g::AbstractArray{T,N};
    incr::Bool = false) where {T,N}

    return call!(a*multiplier(R), func(R), x, g; incr=incr)
end

function call(a::Real,
    R::Regul,
    x::AbstractArray{T,N}) where {T,N}
    
    return call(a*multiplier(R), func(R), x)
end


function get_grad_op(a::Real,
    R::Regul)

    return get_grad_op(a*multiplier(R), func(R))
end

function get_grad_op(R::Regul)

    return get_grad_op(1.0, func(R))
end



"""
    HomogenRegul(Reg, deg)

yields a structure of supertype `Regularization` representing an homogeneous
regularization and containing the multiplier `mu` which tunes it and the
computing process/function which gives its value `f`, in a `Regul` structure,
and the degree of `f`.

Getters of an instance `R` can be imported with `InversePbm.` before:
 - multiplier(R)           # gets the multiplier `mu`.
 - func(R)                 # gets the computing process `f`.
 - param(R)                # gets the inner parameters of the regularization method.
 - degree(R)               # gets the degree `deg`.
 - use_direct_inversion(R) # indicates if it is possible to use the direct
   inversion method to solve the Normal equations.

# Examples
It is possible to create a new instance by directly giving the elements:
```julia
julia> R = HomogenRegul([mu=1,] f [inv=false, deg=degree(f)])
``

Applying a scalar `b` to an insance `R` yields a new instance of `HomogenRegul`:
```julia
julia> R = b*HomogenRegul(mu, f, d, true)
Regul:
 - level `mu` : b*mu
 - function `func` : f
 - use direct inversion `direct_inversion` : true
 - degree `deg` : d
 ```
 
 To apply it to an `AbstractArray` `x`, use:
```julia
julia> R([a,] x)             # apply call([a,] R, x)
julia> R([a,] x, g [; incr]) # apply call!([a,] R, x, g [; incr])
```

See also [`Regul`](@reff)

"""
struct HomogenRegul{T} <: Regularization
    Reg::Regul{T}
    deg::T
end
multiplier(R::HomogenRegul) = multiplier(R.Reg)
func(R::HomogenRegul) = func(R.Reg)
degree(R::HomogenRegul) = R.deg
use_direct_inversion(R::HomogenRegul) = use_direct_inversion(R.Reg)

HomogenRegul(mu::Real, f, inv::Bool, deg::Real) = HomogenRegul(Regul(mu, f, inv), deg)
HomogenRegul(mu::Real, f, inv::Bool=false) = HomogenRegul(mu, f, inv, degree(f))
HomogenRegul(f, inv::Bool=false) = HomogenRegul(1.0, f, inv)

*(a::Real, R::HomogenRegul) = HomogenRegul(a*multiplier(R), func(R), use_direct_inversion(R))

Base.show(io::IO, R::HomogenRegul) = begin
    print(io,"HomogenRegul:")
    print(io,"\n - level `mu` : ",multiplier(R))
    print(io,"\n - function `func` : ",func(R))
    print(io,"\n - use direct inversion `direct_inversion` : ",use_direct_inversion(R))
    print(io,"\n - degree `deg` : ",degree(R))
end

call!(a, R::HomogenRegul, x, g; kwds...) = call!(a, R.Reg, x, g; kwds...)
call!(R::HomogenRegul, x, g; kwds...) = call!(R.Reg, x, g; kwds...)
call(a, R::HomogenRegul, x) = call(a, R.Reg, x)
call(R::HomogenRegul, x) = call(R.Reg, x)
get_grad_op(a, R::HomogenRegul) = get_grad_op(a, R.Reg)
get_grad_op(R::HomogenRegul) = get_grad_op(1.0, R)




"""
    SumRegul(R1, R2)

yields a sum of regularizations that is itself a `Regularization` type. To apply
an array `x` to such a structure, it is possible to use the functions `call!`
and `call` or to use the `LazyAlgebra` operator formalism.

"""
struct SumRegul{Reg<:Regularization,R<:Regularization} <: Regularization
    R1::Reg
    R2::R
end

Base.:+(R1::Reg, R2::Regularization) where {Reg<:Regularization} = SumRegul(R1, R2)

function call!(a, S::SumRegul, x, g; kwds...)
    return call!(a, S.R1, x, g; kwds...) + call!(a, S.R2, x, g; kwds...)
end

function call(a, S::SumRegul, x)
    return call(a, S.R1, x) + call(a, S.R2, x)
end




"""
    InvProblem(L, R)

yields a structure containing the ingredients to compute a sub-problem
criterion for a value of x:
                (A.x - d)'.Diag(w).(A.x - d) + R(x)
with `R` a regularization structure, `A`, `d` and `w` are contained in `L`
(`Lkl` structure).

Getters of an instance `S` can be imported with `InversePbm.` before:
 - likelihood(S) # gets `Lkl` structure.
 - model(S)      # gets the model `A`.
 - data(S)       # gets the data `d`.
 - weights(S)    # gets the weights `w` associated to the data `d`.
 - regul(S)      # gets the `Regularization` `R`.
 - input_size(L)  # gets the input size that should correspond to the input size
   of `A` and to the size of `x`.
 - output_size(L) # gets the output size, that is the same size as `d`.

# Examples
It is possible to create a new instance by directly giving the elements:
```julia
julia> S = InvProblem(A, d, w, R)
```

 To apply it to an `AbstractArray` `x`, use:
```julia
julia> S([a,] x)             # apply call([a,] S, x)
julia> S([a,] x, g [; incr]) # apply call!([a,] S, x, g [; incr])
```
where the second line updates the gradient of the criterion, stored in `g`,
while yielding the result of the first line.

See also: [`Lkl`](@ref), [`Regul`](@ref), [`HomogenRegul`](@ref)

"""
struct InvProblem{RG<:Regularization} <: Cost
    L::Lkl
    R::RG
end

function InvProblem(A::Mapping,
    d::AbstractArray{T,N},
    w::AbstractArray{T,N},
    R::Regularization) where {T,N}

    return InvProblem(Lkl(A, d, w), R)
end

likelihood(S::InvProblem) = S.L
model(S::InvProblem) = model(S.L)
data(S::InvProblem) = data(S.L)
weights(S::InvProblem) = weights(S.L)
regul(S::InvProblem) = S.R
input_size(S::InvProblem) = input_size(model(S))
output_size(S::InvProblem) = size(data(S))


function call!(a::Real,
    S::InvProblem,
    x::AbstractArray{T,N},
    g::AbstractArray{T,N};
    incr::Bool = false) where {T,N}
    
    L, R = likelihood(S), regul(S)
    lkl = L(a, x, g; incr=incr)
    
    return Float64(lkl + R(a, x, g; incr=true))
end

function call(a::Real,
    S::InvProblem,
    x::AbstractArray{T,N}) where {T,N}
    
    L, R = likelihood(S), regul(S)
    
    return Float64(L(a, x) + R(a, x))
end

"""
    BilinearInverseProblem
"""

abstract type BilinearInverseProblem <: Mapping end

(P::BilinearInverseProblem)(::Val{:degJ})=call(P,Val(:degJ))
(P::BilinearInverseProblem)(::Val{:degK})=call(P,Val(:degK))
(P::BilinearInverseProblem)(::Val{:Jx},x)=call(P,Val(:Jx),x)
(P::BilinearInverseProblem)(::Val{:Ky},y)=call(P,Val(:Ky),y)
(P::BilinearInverseProblem)(::Val{:x},x,y,μ)=call(P,Val(:x),x,y,μ)
(P::BilinearInverseProblem)(::Val{:y},x,y,ν)=call(P,Val(:y),x,y,ν)
(P::BilinearInverseProblem)(x,y,μ,ν)=call(P,x,y,μ,ν)

"""
    BilinearProblem
"""
struct BilinearProblem{M1<:Mapping,
                       M2<:Mapping,
                       D<:AbstractArray,
                       R1<:Regularization,
                       R2<:Regularization} <: BilinearInverseProblem
    d::D
    w::D
    Fx::M1
    Fy::M2
    Rx::R1
    Ry::R2
end

call(P::BilinearProblem, ::Val{:degJ})=degree(P.Rx)
call(P::BilinearProblem, ::Val{:degK})=degree(P.Ry)
call(P::BilinearProblem, ::Val{:Jx},x)=call(1.,P.Rx,x)
call(P::BilinearProblem, ::Val{:Ky},y)=call(1.,P.Ry,y)


function call(P::BilinearProblem, ::Val{:x},x::AbstractArray{T,N},y::AbstractVector{T},μ::T) where {T<:AbstractFloat,N}
    Y = P.Fy * y
    yFx = Diag(Y)*P.Fx
    Cx = Lkl(yFx,P.d,P.w)
    Ix = InvProblem(Cx, μ*P.Rx)
    function fg_solve!(x::AbstractArray{T,N}, g::AbstractArray{T,N})
        return call!(Ix, h, g)
    end
    vmlmb!(fg_solve!, x; lower=T(0),maxiter=50, verb=10, mem=3) #FIXME
    return x, call(1., Cx, x), call(μ,P.Rx,x)
end

function call(P::BilinearProblem, ::Val{:y},x::AbstractArray{T,N},y::AbstractVector{T},ν::T) where {T<:AbstractFloat,N}
    X = P.Fx * x
    XFy = Diag(X) * P.Fy
    Cy = Lkl(XFy,P.d,P.w)
    Iy = InvProblem(Cy, ν*P.Ry)
    function fg_solve!(y::AbstractVector{T}, g::AbstractVector{T})
        return call!(Iy, y, g)
    end
    vmlmb!(fg_solve!, y; lower=T(0),maxiter=50, verb=10, mem=3) #FIXME
    return y, call(1., Cy, y), call(ν,P.Ry,y)
end

function call(P::BilinearProblem, x::AbstractArray{T,N},y::AbstractVector{T},μ::T,ν::T) where {T<:AbstractFloat,N}
    X = P.Fx * x
    Y = P.fY * y
    M = X.*Y
    res = P.d - M
    Df = vdot(res, P.w .* res)
    Rx = call(μ,P.Rx,x)
    Ry = call(ν,P.Ry,y)
    return Df + Rx + Ry
end


"""
    Solver

Abstract type shared by methods for solving inverse problems. The type is used
to specified which method to use when optimizing on a `InvProblem` structure. An
instance of a structure of super-type `Solver` obeys to solving methods `solve`
and `solve!`.

"""
abstract type Solver end

"""
    solve!([x=ones(input_size(S)),] S, m [,c=[]]; kwds...)

yields the solution of applying the optimization method `m` on the criterion `S`
and store it in the initialization `x`. If `x` is not given, a one
`AbstractArray` of same size as the input argument of `S` is created to
initialize the optimization method `m`. `c` can contains the values of the
criterion computed for optimization by the method `m`.

# Keywords
 - keep_loss : (`false` by default) allows to store the values of the criterion
   in the cache `c`.

"""
solve!(S, m, c=Float64[]; kwds...) = solve!(ones(input_size(S)), S, m, c; kwds...)

"""
    solve([x=ones(input_size(S)),] S, m [,c=[]]; kwds...)

yields the solution of applying the optimization method `m` on the problem `S`.
If `x` is not given, a one `AbstractArray` of same size as the input argument of
`S` is created to initialize the optimization method `m`. `c` can contains the
values of the criterion computed for optimization by the method `m`.

# Keywords
 - keep_loss : (`false` by default) allows to store the values of the criterion
   in the cache `c`.

"""
solve(x::AbstractArray, S, m, c=Float64[]; kwds...) = solve!(vcopy(x), S, m, c; kwds...)
solve(S, m, c=Float64[]; kwds...) = solve(ones(input_size(S)), S, m, c; kwds...)

