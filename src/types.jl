#
# types.jl --
#
# Defines the generic structures necessary for the module InverseProblem
#
#------------------------------------------------------------------------------
#


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

#FIXME: add the call to a sum of :Cost

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
    Regularization

Abstract type which obeys the `call!` and `call` laws, and behaves as a `Cost` type. 

"""
abstract type Regularization <: Cost end




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

"""
struct Regul{T<:Real,F} <: Regularization
    mu::T # multiplier
    f::F # function
end
multiplier(R::Regul) = R.mu
func(R::Regul) = R.f

Regul(f) = Regul(1.0, f)

*(a::Real, R::Regul) = Regul(a*multiplier(R), func(R))
Base.show(io::IO, R::Regul) = begin
    print(io,"Regul:")
    print(io,"\n - level `mu` : ",multiplier(R))
    print(io,"\n - function `func` : ",func(R))
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

HomogenRegul(mu::Real, f, deg::Real) = HomogenRegul(Regul(mu, f), deg)
HomogenRegul(mu::Real, f) = HomogenRegul(mu, f, degree(f))
HomogenRegul(f) = HomogenRegul(1.0, f)

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
struct Lkl{F<:Function,D<:AbstractArray} <: Cost
    A::F # model
    d::D # data
    w::D # weights of data

    function Lkl(A::F, d::D, w::D) where {F<:Function,D<:AbstractArray}
        @assert size(d) == size(w)# == output_size(A)
        return new{F,D}(A, d, w)
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

    @assert typeof(A) <: Mapping #FIXME: if not, not able to update gradient g

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
    SubProblem(L, R)

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
julia> S = SubProblem(A, d, w, R)
```

 To apply it to an `AbstractArray` `x`, use:
```julia
julia> S([a,] x)             # apply call([a,] S, x)
julia> S([a,] x, g [; incr]) # apply call!([a,] S, x, g [; incr])
```

"""
struct SubProblem{RG<:Regularization} <: Cost
    L::Lkl
    R::RG
end

function SubProblem(A,
    d::AbstractArray{T,N},
    w::AbstractArray{T,N},
    R::Regularization) where {T,N}

    return SubProblem(Lkl(A, d, w), R)
end

likelihood(S::SubProblem) = S.L
model(S::SubProblem) = model(S.L)
data(S::SubProblem) = data(S.L)
weights(S::SubProblem) = weights(S.L)
regul(S::SubProblem) = S.R
input_size(S::SubProblem) = input_size(model(S))
output_size(S::SubProblem) = size(data(S))


function call!(a::Real,
    S::SubProblem,
    x::AbstractArray{T,N},
    g::AbstractArray{T,N};
    incr::Bool = false) where {T,N}
    
    L, R = likelihood(S), regul(S)
    lkl = L(a, x, g; incr=incr)
    
    return Float64(lkl + R(a, x, g; incr=true))
end

function call(a::Real,
    S::SubProblem,
    x::AbstractArray{T,N}) where {T,N}
    
    L, R = likelihood(S), regul(S)
    
    return Float64(L(a, x) + R(a, x))
end




"""
    Solver

Abstract type shared by methods for solving inverse problems. The type is used to specified
which method to use when optimizing on a `SubProblem` structure. An instance of a structure 
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


