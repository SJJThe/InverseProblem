#
# regularization.jl
#
# Defines different regularization methods.
#
#------------------------------------------------------------------------------
#
# This file is part of InverseProblem

"""
    norml1

yields an instance of the L1 norm, that is for an array `x`
```
                  ||x||_1 = ∑ |x_i|
                            i
```
If the `call!` method is used, it suppose that `x` is positif.

`norml1` is a `HomogenRegul` structure of degree `1`.

# Example
```julia
julia> R = mu*norml1()
julia> R([a,] x)            # apply call([a,] R, x)
julia> R([a,] x, g [; incr]) # apply call!([a,] R, x, g [; incr])
```

See also [`HomogenRegul`](@ref)

"""
struct NormL1 <: Regularization end

function call!(a::Real,
    ::NormL1,
    x::AbstractArray{T,N},
    g::AbstractArray{T,N};
    incr::Bool = false) where {T,N}

    @assert !any(e -> e < T(0), x)
    !incr && vfill!(g, 0.0)
    f::Float64 = 0
    @inbounds @simd for i in eachindex(x, g)
        f += x[i]
        g[i] = Float64(a)
    end

    return a*f
end

function call(a::Real,
    ::NormL1,
    x::AbstractArray{T,N}) where {T,N}

    f::Float64 = 0
    @inbounds @simd for i in eachindex(x)
        f += abs(x[i])
    end

    return a*f
end

function get_grad_op(a::Real,
    ::NormL1)

    return Float64(a)
end

degree(::NormL1) = 1.0

norml1() = HomogenRegul(NormL1(), true)




"""
    norml2

yields an instance of the L2 norm, i.e.
```
                  ||x||_2^2 = ∑ (x_i)^2
                              i
```

`norml2` is a `HomogenRegul` structure of degree `2`.

# Example
```julia
julia> R = mu*norml2()
julia> R([a,] x)            # apply call([a,] R, x)
julia> R([a,] x, g [; incr]) # apply call!([a,] R, x, g [; incr])
```

See also [`HomogenRegul`](@ref)

"""
struct NormL2 <: Regularization end

function call!(a::Real,
    ::NormL2,
    x::AbstractArray{T,N},
    g::AbstractArray{T,N};
    incr::Bool = false) where {T,N}

    !incr && vfill!(g, 0.0)
    f::Float64 = 0
    @inbounds @simd for i in eachindex(x, g)
        f += x[i]^2
        g[i] = 2*Float64(a)*x[i]
    end

    return a*f
end

function call(a::Real,
    ::NormL2,
    x::AbstractArray{T,N}) where {T,N}

    f::Float64 = 0
    @inbounds @simd for i in eachindex(x)
        f += x[i]^2
    end

    return a*f
end

function get_grad_op(a::Real,
    ::NormL2)

    return 2*Float64(a)
end

degree(::NormL2) = 2.0

norml2() = HomogenRegul(NormL2(), true)




"""
    tikhonov

yields an instance of Tikhonov smoothness regularization of `x`, that is:
```
                    ∑ ||D_i*x||_2^2
                    i
```
with `D_i`, an approximation of the gradient of `x` at index `i`.

`tikhonov` is a `HomogenRegul` structure of degree `2`.

# Example
```julia
julia> R = mu*tikhonov()
julia> R([a,] x)            # apply call([a,] R, x)
julia> R([a,] x, g [; incr]) # apply call!([a,] R, x, g [; incr])
```

See also [`HomogenRegul`](@ref)

"""
struct Tikhonov <: Regularization end

function call!(a::Real,
    ::Tikhonov,
    x::AbstractArray{T,N},
    g::AbstractArray{T,N};
    incr::Bool = false) where {T,N}
    
    D = Diff()
    apply!(2*a, D'*D, x, (incr ? 1 : 0), g)

    return Float64(a*vnorm2(D*x)^2)
end

function call(a::Real,
    ::Tikhonov,
    x::AbstractArray{T,N}) where {T,N}
    
    D = Diff()

    return Float64(a*vnorm2(D*x)^2)
end

function get_grad_op(a::Real,
    ::Tikhonov)

    D = Diff()
    
    return 2*Float64(a)*(D'*D)
end

degree(::Tikhonov) = 2.0

tikhonov() = HomogenRegul(Tikhonov(), true)


"""
    WeightedTikhonov

    yields an instance of weighted Tikhonov smoothness regularization of `x`, that is:
```
                    ∑ ||D_i x||_w^2
                    i
```
with `D_i`, an approximation of the gradient of `x` at index `i`.

`tikhonov` is a `HomogenRegul` structure of degree `2`.

# Example
```julia
julia> R = WeightedTikhonov(mu, w)
julia> R([a,] x)            # apply call([a,] R, x)
julia> R([a,] x, g [; incr]) # apply call!([a,] R, x, g [; incr])
```

"""
struct WeightedTikhonov{T<:Real} <: Regularization
    mu::T # multiplier
    w::AbstractArray
    
    function WeightedTikhonov(mu::T,w::AbstractArray) where {T<:Real}
        return new{T}(mu,w)
    end
end

multiplier(R::WeightedTikhonov) = R.mu
weights(R::WeightedTikhonov) = R.w

#WeightedTikhonov(w) = WeightedTikhonov(1.0, w)
#WeightedTikhonov(mu::Real) = mu*tikhonov()

*(a::Real, R::WeightedTikhonov) = WeightedTikhonov(a*multiplier(R), weights(R))

Base.show(io::IO, R::WeightedTikhonov) = begin
    print(io,"Weighted Tikhonov:")
    print(io,"\n - level `mu` : ",multiplier(R))
    print(io,"\n - weights` : ", weights(R))
end

function call!(a::Real,
    R::WeightedTikhonov,
    x::AbstractArray{T,N},
    g::AbstractArray{T,N};
    incr::Bool = false) where {T,N}

    W = weights(R)
    mu = multiplier(R)
    @assert size(x) == size(W)[1:end-1] && size(W)[end] == N
    
    D = Diff()
    r = D*x
    wr = Diag(W)*r
        
    apply!(2*a*mu, D', wr, (incr ? 1 : 0), g)
    
    return Float64(a*mu*vdot(r, wr))
end

function call(a::Real,
    R::WeightedTikhonov,
    x::AbstractArray{T,N}) where {T,N}

    W = weights(R)
    mu = multiplier(R)
    @assert size(x) == size(W)[1:end-1] && size(W)[end] == N
    
    D = Diff()
    
    r= D*x
    wr= Diag(W)*r
 
    return Float64(mu*a*vdot(r, wr))
end


degree(::WeightedTikhonov) = 2.0



"""
    edgepreserving(τ [, v=:v1])

builds an `EdgePreserving` `Regul` structure, containing an intern tuning 
parameter `τ`. Used as an operator on an `AbstractArray` `x`, an instance `R`
of `EdgePreserving` will give back:
```
            R(x) = 2*ρ*∑ ( sqrt( ||D_i.x||_2^2 + ρ^2 ) - ρ )
                       i
``
with `D_i`, an approximation of the gradient of `x` at index `i` and `ρ = √2 *τ`.

Getter of an instance `R` can be imported with `InversePbm.` before:
 - param(R) # gets the parameter `τ`.

The `v` optional argument indicates the version of the regularization used. It is
solely specified here in case of another formulation of the regularization is 
wanted and implemented.

# Example
```julia
julia> R = mu*edgepreserving(τ)
julia> R([a,] x)            # apply call([a,] R, x)
julia> R([a,] x, g [; incr]) # apply call!([a,] R, x, g [; incr])
```

See also [`Regul`](@ref)

"""
struct EdgePreserving{V,T<:AbstractFloat} <: Regularization 
    τ::T
end
param(R::EdgePreserving) = R.τ

function call!(a::Real,
    R::EdgePreserving{:v1,T},
    x::AbstractArray{T,2},
    g::AbstractArray{T,2};
    incr::Bool = false) where {T}

    n1, n2 = size(x)

    @assert n1 == size(g, 1) && n2 == size(g, 2)

    τ = param(R)
    ρ = sqrt(2)*τ
    ρ2 = ρ^2
    α = 2*ρ
    cf = T(a*α)

    !incr && vfill!(g, 0.0)
    f_ij::Float64 = 0
    @inbounds for j in 1:n2
        jp1 = min(j+1, n2)
        @simd for i in 1:n1
            ip1 = min(i+1, n1)
            d1 = x[ip1,j] - x[i,j]
            d2 = x[i,jp1] - x[i,j]
            r_ij = sqrt(d1^2 + d2^2 + ρ2)
            f_ij += r_ij

            q_ij = 1.0/r_ij
            cf_ij = cf*q_ij
            g[i,j] -= cf_ij*(d1 + d2)
            g[ip1,j] += cf_ij*d1
            g[i,jp1] += cf_ij*d2
        end
    end

    return cf*(f_ij - ρ)
end

function call(a::Real,
    R::EdgePreserving{:v1,T},
    x::AbstractArray{T,2}) where {T}

    n1, n2 = size(x)

    τ = param(R)
    ρ = sqrt(2)*τ
    ρ2 = ρ^2
    α = 2*ρ
    cf = T(a*α)

    f::Float64 = 0
    @inbounds for j in 1:n2
        jp1 = min(j+1, n2)
        @simd for i in 1:n1
            ip1 = min(i+1, n1)
            d1 = x[ip1,j] - x[i,j]
            d2 = x[i,jp1] - x[i,j]
            f += sqrt(d1^2 + d2^2 + ρ2)
        end
    end
    
    return cf*(f - ρ)
end

edgepreserving(e::T, V::Symbol = :v1) where {T} = Regul(EdgePreserving{V,Float64}(Float64(e)), false)




"""
    homogenedgepreserving(τ [, v=:v1])

builds an `HomogenEdgePreserving` `HomogenRegul` structure, containing an intern
tuning parameter `τ`. Used as an operator on an `AbstractArray` `x`, an instance
`R` of `HomogenEdgePreserving` will give back:
```
        R(x) = 2*ρ*||x||_2^β*∑( √(||D_i*x||_2^2 + ρ^2*||x||_2^2) - ρ*||x||_2 )
                             i
```
with `ρ` = √(2/N)*τ, `β` = 1 and `D_i`, an approximation of the gradient of `x`
at index `i`.

Getter of an instance `R` can be imported with `InversePbm.` before:
 - param(R) # gets the parameter `τ`.

The `v` optional argument indicates the version of the regularization used. It is
solely specified here in case of another formulation of the regularization is 
wanted and implemented.

# Example
```julia
julia> R = mu*homogenedgepreserving(τ)
julia> R([a,] x)            # apply call([a,] R, x)
julia> R([a,] x, g [; incr]) # apply call!([a,] R, x, g [; incr])
```

See also [`HomogenRegul`](@ref)

"""
struct HomogenEdgePreserving{V,T<:Real} <: Regularization 
    τ::T
end
param(R::HomogenEdgePreserving) = R.τ

function call!(a::Real,
    R::HomogenEdgePreserving{:v1,T},
    x::AbstractArray{T,2},
    g::AbstractArray{T,2};
    incr::Bool = false) where {T}
    
    N1, N2 = size(x)
    @assert N1 == size(g, 1) && N2 == size(g, 2)

    τ = param(R)
    ρ = sqrt(2/(N1*N2))*τ 
    ρ2 = ρ^2
    α = 2*ρ
    β = 1.0
    norm_x2 = vdot(x,x)
    ρ2normx2 = ρ2*norm_x2
    norm_x = sqrt(norm_x2)
    ρnorm_x = ρ*norm_x
    norm_xβm2 = norm_x^(β-2)
    norm_xβ = norm_xβm2*norm_x2
    cf = T(a*α*norm_xβ)

    !incr && vfill!(g, 0.0)
    f = Float64(0)
    t = Float64(0)
    @inbounds for j in 1:N2
        jp1 = min(j+1, N2)
        @simd for i in 1:N1
            ip1 = min(i+1, N1)
            # finite differences
            d1_ij = (x[ip1,j] - x[i,j])
            d2_ij = (x[i,jp1] - x[i,j])
            r_ij = sqrt(d1_ij^2 + d2_ij^2 + ρ2normx2)
            f += r_ij

            q_ij = 1/(2*r_ij)
            t += q_ij
            g[i,j] -= 2*cf*q_ij*(d1_ij + d2_ij)
            g[ip1,j] += 2*cf*q_ij*(x[ip1,j] - x[i,j])
            g[i,jp1] += 2*cf*q_ij*(x[i,jp1] - x[i,j])
        end
    end
    u = T(a*α*norm_xβm2*(β*f - N1*N2*(1+β)*ρnorm_x + 2*ρ2normx2*t))
    @inbounds @simd for i in eachindex(x, g)
        g[i] += u*x[i]
    end

    return cf*(f - N1*N2*ρnorm_x)
end

function call(a::Real,
    R::HomogenEdgePreserving{:v1,T},
    x::AbstractArray{T,2}) where {T}

    τ = param(R)
    N1, N2 = size(x)
    ρ = sqrt(2/(N1*N2))*τ 
    ρ2 = ρ^2
    α = 2*ρ
    β = 1.0
    norm_x2 = vdot(x,x)
    ρ2normx2 = ρ2*norm_x2
    norm_x = sqrt(norm_x2)
    cf = T(a*α*norm_x^β)
    f = Float64(0)
    @inbounds for j in 1:N2
        jp1 = min(j+1, N2)
        @simd for i in 1:N1
            ip1 = min(i+1, N1)
            d1 = (x[ip1,j] - x[i,j])
            d2 = (x[i,jp1] - x[i,j])
            f += sqrt(d1^2 + d2^2 + ρ2normx2)
        end
    end

    return cf*(f - N1*N2*ρ*norm_x)
end

degree(::HomogenEdgePreserving{:v1}) = 2.0


homogenedgepreserving(e::T, V::Symbol = :v1) where {T} = HomogenRegul(HomogenEdgePreserving{V,T}(e))


