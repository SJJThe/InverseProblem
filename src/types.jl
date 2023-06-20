#
# types.jl --
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

"""
abstract type Cost <: Mapping end

(C::Cost)(a, x) = call(a, C, x)
(C::Cost)(x) = call(1.0, C, x)
(C::Cost)(a, x::D, g::D; kwds...) where {D} = call!(a, C, x, g; kwds...)
(C::Cost)(x::D, g::D; kwds...) where {D} = call!(1.0, C, x, g; kwds...)



"""
    call!([a::Real=1,] f, x, g; incr::Bool = false)

gives back the value of a*f(x) while updating (in place) its gradient in g. 
incr is a boolean indicating if the gradient needs to be incremanted or 
reseted.

"""
call!(f, x, g; kwds...) = call!(1.0, f, x, g; kwds...)

"""
    call([a::Real=1,] f, x)

yields the value of a*f(x).

"""
call(f, x) = call(1.0, f, x)




"""
    Lkl(A, d [,w=ones(size(d))])

yields a structure containing the elements essential to compute the Maximum likelihood 
criterion, in the Gaussian hypothesis, of an `AbstractArray` `x`, that is:
                            (A(x) - d)'.Diag(w).(A(x) - d)

Getters of an instance `L` can be imported with `InversePbm.` before:
 - model(L)   # gets the model `A`.
 - data(L)    # gets the data `d`.
 - weights(L) # gets the weights `w` associated to the data `d`.

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
        @assert size(d) == size(w)# == output_size(A)
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


#FIXME: a sum of Regularization must be a Regularization


"""
    Regul([mu::Real=1,] f)

yields a structure of supertype `Regularization` containing the element necessary 
to compute a regularization term `f` with it's multiplier `mu`. Creating an instance 
of Regul permits to use the `call` and `call!` functions specialized.

Getters of an instance `R` can be imported with `InversePbm.` before:
 - multiplier(R) # gets the multiplier `mu`.
 - func(R)       # gets the computing process `f`.

# Example
```julia
julia> R = Regul(a, MyRegul)
Regul:
 - level `mu` : a
 - function `func` : MyRegul
```
with `MyRegul` a strucure which defines the behaviors of `call([a,] MyRegul, x)` and 
`call!([a,] MyRegul, x, g [;incr])` on an `AbstractArray` `x` and its gradient `g`.

To apply it to `x`, use:
```julia
julia> R([a,] x)            # apply call([a,] R, x)
julia> R([a,] x, g [;incr]) # apply call!([a,] R, x, g [;incr])
```

Applying a scalar `b` to an insance `R` yields a new instance of `Regul`:
```julia
julia> R = b*Regul(a, MyRegul)
Regul:
 - level `mu` : b*a
 - function `func` : MyRegul
```

#FIXME: update

"""
struct Regul{T<:Real,F} <: Regularization
    mu::T # multiplier
    f::F # function
    direct_inversion::Bool
end
multiplier(R::Regul) = R.mu
func(R::Regul) = R.f
use_direct_inversion(R::Regul) = R.direct_inversion

Regul(f) = Regul(1.0, f)
Regul(f, i::Bool) = Regul(1.0, f, i)

*(a::Real, R::Regul) = Regul(a*multiplier(R), func(R))

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

function call!(R::Regul,
    x::AbstractArray{T,N},
    g::AbstractArray{T,N};
    incr::Bool = false) where {T,N}

    return call!(multiplier(R), func(R), x, g; incr=incr)
end

function call(a::Real,
    R::Regul,
    x::AbstractArray{T,N}) where {T,N}
    
    return call(a*multiplier(R), func(R), x)
end

function call(R::Regul,
    x::AbstractArray{T,N}) where {T,N}
    
    return call(multiplier(R), func(R), x)
end

function get_grad_op(a::Real,
    R::Regul)

    return get_grad_op(a*multiplier(R), func(R))
end

function get_grad_op(R::Regul)

    return get_grad_op(multiplier(R), func(R))
end




"""
    HomogenRegul(Reg, deg)

yields a structure of supertype `Regularization` representing an homogeneous regularization 
and containing the multiplier `mu` which tunes it and the computing process/function
which gives its value `f`, in a `Regul` structure, and the degree of `f`.

Getters of an instance `R` can be imported with `InversePbm.` before:
 - multiplier(R) # gets the multiplier `mu`.
 - func(R)       # gets the computing process `f`.
 - param(R)      # gets the inner parameters of the regularization method.
 - degree(R)     # gets the degree `deg`.

# Examples
It is possible to create a new instance by directly giving the elements:
```julia
julia> R = HomogenRegul([mu=1,] f [,deg=degree(f)])
``

Applying a scalar `b` to an insance `R` yields a new instance of `HomogenRegul`:
```julia
julia> R = b*HomogenRegul(mu, f, d)
Regul:
 - level `mu` : b*mu
 - function `func` : f
 - degree `deg` : d
 ```
 
 To apply it to an `AbstractArray` `x`, use:
```julia
julia> R([a,] x)             # apply call([a,] R, x)
julia> R([a,] x, g [; incr]) # apply call!([a,] R, x, g [; incr])
```

"""
struct HomogenRegul{T} <: Regularization
    Reg::Regul{T}
    deg::T
end
multiplier(R::HomogenRegul) = multiplier(R.Reg)
func(R::HomogenRegul) = func(R.Reg)
degree(R::HomogenRegul) = R.deg

HomogenRegul(mu::Real, f, inv::Bool, deg::Real) = HomogenRegul(Regul(mu, f, inv), deg)
HomogenRegul(mu::Real, f, inv::Bool=false) = HomogenRegul(mu, f, inv, degree(f))
HomogenRegul(f, inv::Bool=false) = HomogenRegul(1.0, f, inv)

*(a::Real, R::HomogenRegul) = HomogenRegul(a*multiplier(R), func(R))
Base.show(io::IO, R::HomogenRegul) = begin
    print(io,"HomogenRegul:")
    print(io,"\n - level `mu` : ",multiplier(R))
    print(io,"\n - function `func` : ",func(R))
    print(io,"\n - degree `deg` : ",degree(R))
end

call!(a, R::HomogenRegul, x, g; kwds...) = call!(a, R.Reg, x, g; kwds...)
call!(R::HomogenRegul, x, g; kwds...) = call!(R.Reg, x, g; kwds...)
call(a, R::HomogenRegul, x) = call(a, R.Reg, x)
call(R::HomogenRegul, x) = call(R.Reg, x)




"""
    InvProblem(L, R)

yields a structure containing the ingredients to compute a sub-problem
criterion for a value of x:
                (A.x - d)'.Diag(w).(A.x - d) + R(x)
with `R` a regularization structure, `A`, `d` and `w` are contained in `L` (`Lkl` structure).

Getters of an instance `S` can be imported with `InversePbm.` before:
 - likelihood(S) # gets `Lkl` structure.
 - model(S)      # gets the model `A`.
 - data(S)       # gets the data `d`.
 - weights(S)    # gets the weights `w` associated to the data `d`.
 - regul(S)      # gets the `Regularization` `R`.

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
    Solver

Abstract type shared by methods for solving inverse problems. The type is used to specified
which method to use when optimizing on a `InvProblem` structure. An instance of a structure 
of super-type `Solver` obeys to solving methods `solve` and `solve!`. 

"""
abstract type Solver end

"""
    solve!([x::AbstractArray,] S, m [,c=[]]; kwds...)

yields the solution of applying the optimization method `m` on the problem `S` and store it
in the initialization `x`. If `x` is not given, a one `AbstractArray` of same size as the 
input argument of `S` is created to initialize the optimization method `m`. `c` can contains 
the values of the criterion computed for optimization by the method `m`.

# Keywords
 - keep_loss : (`false` by default) allows to store the values of the criterion in `cache`.

"""
solve!(S, m, c=Float64[]; kwds...) = solve!(ones(input_size(S)), S, m, c; kwds...)

"""
    solve([x::AbstractArray,] S, m [,c=[]]; kwds...)

yields the solution of applying the optimization method `m` on the problem `S`. If `x` is 
not given, a one `AbstractArray` of same size as the input argument of `S` is created to 
initialize the optimization method `m`. `c` can contains the values of the criterion 
computed for optimization by the method `m`.

# Keywords
 - keep_loss : (`false` by default) allows to store the values of the criterion in `cache`.

"""
solve(x::AbstractArray, S, m, c=Float64[]; kwds...) = solve!(vcopy(x), S, m, c; kwds...)
solve(S, m, c=Float64[]; kwds...) = solve(ones(input_size(S)), S, m, c; kwds...)


