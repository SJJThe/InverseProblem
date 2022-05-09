#
# regularization.jl --
#
# Defines different regularization methods for the module InverseProblem.
#
#------------------------------------------------------------------------------
#


"""
    edgepreserving(τ)

builds an `EdgePreserving` `Regul` structure, containing an intern tuning 
parameter `τ`. Used as an operator on an `AbstractArray` `x`, an instance `R` 
of `EdgePreserving` will give back:
```
                    R(x) = ∑ sqrt( ||D_i.x||_2^2 + τ^2 )
                           i
``
with `D_i`, an approximation of the gradient of `x` at index `i`.

Getter of an instance `R` can be imported with `InversePbm.` before:
 - param(R) # gets the parameter `τ`.

# Example
```julia
julia> R = mu*edgepreserving(τ)
julia> R([a,] x)            # apply call([a,] R, x)
julia> R([a,] x, g [;incr]) # apply call!([a,] R, x, g [;incr])
```

"""
struct EdgePreserving{V,T<:Real} <: Regularization 
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

    eps2 = param(R)[1]^2
    f::Float64 = 0
    t::Float64 = 0
    !incr && vfill!(g, 0.0)
    @inbounds for j in 1:n2
        jp1 = min(j+1, n2)
        @simd for i in 1:n1
            ip1 = min(i+1, n1)
            d1 = x[ip1,j] - x[i,j]
            d2 = x[i,jp1] - x[i,j]
            r = sqrt(d1^2 + d2^2 + eps2)
            f += r
            q = 1.0/r
            t += q
            d1 *= q
            d2 *= q
            g[i,j] -= d1 + d2
            g[ip1,j] += d1
            g[i,jp1] += d2
        end
    end
    @. g *= a

    return a*f
end

function call(a::Real,
    R::EdgePreserving{:v1,T},
    x::AbstractArray{T,2}) where {T}

    n1, n2 = size(x)
    eps2 = param(R)[1]^2
    f::Float64 = 0
    @inbounds for j in 1:n2
        jp1 = min(j+1, n2)
        @simd for i in 1:n1
            ip1 = min(i+1, n1)
            d1 = x[ip1,j] - x[i,j]
            d2 = x[i,jp1] - x[i,j]
            f += sqrt(d1^2 + d2^2 + eps2)
        end
    end
    
    return a*f
end

#FIXME: code version with automatic tuning of scaling (leveling)
function call!(a::Real,
    R::EdgePreserving{:v2,T},
    x::AbstractArray{T,2},
    g::AbstractArray{T,2};
    incr::Bool = false) where {T}

    n1, n2 = size(x)
    @assert n1 == size(g, 1) && n2 == size(g, 2)

    τ = param(R)
    ρ = sqrt(2/(n1*n2))*τ
    ρ2 = ρ^2
    α = 2*ρ
    cf = T(a*α)

    !incr && vfill!(g, 0.0)
    f = Float64(0)
    @inbounds for j in 1:n2
        jp = min(j+1, n2)
        @simd for i in 1:n1
            ip = min(i+1, n1)
            # finite differences on average over a 2x2 pixels' 
            #interpolated cell
            d_ipjp_ij = x[ip,jp] - x[i,j]
            d_ijp_ipj = x[i,jp] - x[ip,j]
            s_ipj_ijp = x[ip,j] + x[i,jp]
            s_ij_ipjp = x[i,j] + x[ip,jp]
            d_s = s_ij_ipjp - s_ipj_ijp
            Dx2_ij = 1/2*d_ijp_ipj^2 + 1/2*d_ipjp_ij^2 + 
                      1/6*d_s^2

            r_ij = sqrt(Dx2_ij + ρ2)
            f += r_ij

            q_ij = 1/r_ij
            cf_ij = cf/2*q_ij

            g[i,j] += cf_ij*(1/3*d_s - d_ipjp_ij)
            g[ip,j] -= cf_ij*(d_ijp_ipj + 1/3*d_s)
            g[i,jp] += cf_ij*(d_ijp_ipj - 1/3*d_s)
            g[ip,jp] += cf_ij*(d_ipjp_ij + 1/3*d_s)
        end
    end

    return cf*(f - n1*n2*ρ)
end

function call(a::Real,
    R::EdgePreserving{:v2,T},
    x::AbstractArray{T,2}) where {T}

    n1, n2 = size(x)

    τ = param(R)
    ρ = sqrt(2/(n1*n2))*τ
    ρ2 = ρ^2
    α = 2*ρ

    f = Float64(0)
    @inbounds for j in 1:n2
        jp = min(j+1, n2)
        @simd for i in 1:n1
            ip = min(i+1, n1)
            # finite differences on average over a 2x2 pixels' 
            #interpolated cell
            d_ipjp_ij = x[ip,jp] - x[i,j]
            d_ijp_ipj = x[i,jp] - x[ip,j]
            s_ipj_ijp = x[ip,j] + x[i,jp]
            s_ij_ipjp = x[i,j] + x[ip,jp]
            d_s = s_ij_ipjp - s_ipj_ijp
            Dx2_ij = 1/2*d_ijp_ipj^2 + 1/2*d_ipjp_ij^2 + 
                      1/6*d_s^2

            r_ij = sqrt(Dx2_ij + ρ2)
            f += r_ij
        end
    end

    return a*α*(f - n1*n2*ρ)
end

edgepreserving(e::T, V::Symbol = :v1) where {T} = Regul(EdgePreserving{V,T}(e))




"""
    norml1

yields an instance of the L1 norm, that is for an array `x`
```
                  ||x||_1 = ∑ |x_i|
                            i
```
If the `call!` method is used, it suppose that `x` is positif.

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

degree(::NormL1) = 1.0

const norml1 = HomogenRegul(NormL1())




"""
    norml2

yields an instance of the L2 norm, i.e.
```
                  ||x||_2^2 = ∑ (x_i)^2
                              i
```

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

degree(::NormL2) = 2.0

const norml2 = HomogenRegul(NormL2())




"""
    quadraticsmoothness

yields an instance of Tikhonov smoothness regularization of `x`, that is:
```
                    ∑ ||D_i*x||_2^2
                    i
```
with `D_i`, an approximation of the gradient of `x` at index `i`.

"""
struct QuadraticSmoothness <: Regularization end

function call!(a::Real,
    ::QuadraticSmoothness,
    x::AbstractArray{T,N},
    g::AbstractArray{T,N};
    incr::Bool = false) where {T,N}
    
    D = Diff()
    apply!(2*a, D'*D, x, (incr ? 1 : 0), g)

    return Float64(a*vnorm2(D*x)^2)
end

function call(a::Real,
    ::QuadraticSmoothness,
    x::AbstractArray{T,N}) where {T,N}
    
    D = Diff()

    return Float64(a*vnorm2(D*x)^2)
end

degree(::QuadraticSmoothness) = 2.0

const quadraticsmoothness = HomogenRegul(QuadraticSmoothness())




"""
#TODO: code
"""
struct TotalVariation <: Regularization end


function call!(a::Real,
    ::TotalVariation,
    x::AbstractArray{T,N},
    g::AbstractArray{T,N};
    incr::Bool = false) where {T,N}
    
    return 
end

function call(a::Real,
    ::TotalVariation,
    x::AbstractArray{T,N}) where {T,N}
    
    return 
end

degree(::TotalVariation) = 2.0

const totalvariation = HomogenRegul(TotalVariation())




"""
    homogenedgepreserving(τ [, V])

yields an instance of homogeneous edge preserving regularization. Two versions 
are available (`V = :v1` or `V = :v2`). By default, the version 2 is used, 
corresponding for an array `x` ∈ R^N to:
```
        2*ρ*||x||_2^2*∑( √(||D_i*x||_2^2 + ρ^2*||x||_2^2) - ρ*||x||_2 )
                                  i
```
with `ρ` = √(2/N)*τ

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
    
    n1, n2 = size(x)
    
    @assert n1 == size(g, 1) && n2 == size(g, 2)

    eps2 = (param(R)[1])^2/length(x)
    u = eps2*vnorm2(x)^2
    f::Float64 = 0
    t::Float64 = 0
    !incr && vfill!(g, 0.0)
    @inbounds for j in 1:n2
        jp1 = min(j+1, n2)
        @simd for i in 1:n1
            ip1 = min(i+1, n1)
            d1 = x[ip1,j] - x[i,j]
            d2 = x[i,jp1] - x[i,j]
            r = sqrt(d1^2 + d2^2 + u)
            f += r
            q = 1.0/r
            t += q
            d1 *= a*q
            d2 *= a*q
            g[i,j] -= d1 + d2
            g[ip1,j] += d1
            g[i,jp1] += d2
        end
    end
    c = T(a*t*eps2)
    @inbounds @simd for i in eachindex(x, g)
        g[i] += c*x[i]
    end

    return a*f
end

function call(a::Real,
    R::HomogenEdgePreserving{:v1,T},
    x::AbstractArray{T,2}) where {T}

    n1, n2 = size(x)
    eps2 = (param(R)[1])^2/length(x)
    u = eps2*vnorm2(x)^2
    f::Float64 = 0
    @inbounds for j in 1:n2
        jp1 = min(j+1, n2)
        @simd for i in 1:n1
            ip1 = min(i+1, n1)
            d1 = x[ip1,j] - x[i,j]
            d2 = x[i,jp1] - x[i,j]
            f += sqrt(d1^2 + d2^2 + u)
        end
    end
    
    return a*f
end

function call!(a::Real,
    R::HomogenEdgePreserving{:v2,T},
    x::AbstractArray{T,2},
    g::AbstractArray{T,2};
    incr::Bool = false) where {T}
    
    N1, N2 = size(x)
    @assert N1 == size(g, 1) && N2 == size(g, 2)

    τ = param(R)
    ρ = sqrt(2/(N1*N2))*τ 
    ρ2 = ρ^2
    α = 2*ρ
    β = 1
    norm_x2 = vnorm2(x)
    norm_x = sqrt(norm_x2)
    cf = T(a*α*norm_x^β)
    cg = T(a*α*norm_x^(β-2))

    !incr && vfill!(g, 0.0)
    f = Float64(0)
    d = Float64(0)
    @inbounds for j in 1:N2
        jp = min(j+1, N2)
        @simd for i in 1:N1
            ip = min(i+1, N1)
            # finite differences on average over a 2x2 pixels' 
            #interpolated cell
            d_ipjp_ij = x[ip,jp] - x[i,j]
            d_ijp_ipj = x[i,jp] - x[ip,j]
            s_ipj_ijp = x[ip,j] + x[i,jp]
            s_ij_ipjp = x[i,j] + x[ip,jp]
            d_s = s_ij_ipjp - s_ipj_ijp
            Dx2_ij = 1/2*d_ijp_ipj^2 + 1/2*d_ipjp_ij^2 + 
                      1/6*d_s^2
            
            r_ij = sqrt(Dx2_ij + ρ2*norm_x2)
            f += r_ij

            q_ij = 1/r_ij
            cf_ij = cf/2*q_ij
            d += β*r_ij + ρ2*norm_x2*q_ij

            g[i,j] += cf_ij*(1/3*d_s - d_ipjp_ij)
            g[ip,j] -= cf_ij*(d_ijp_ipj + 1/3*d_s)
            g[i,jp] += cf_ij*(d_ijp_ipj - 1/3*d_s)
            g[ip,jp] += cf_ij*(d_ipjp_ij + 1/3*d_s)
        end
    end
    d -= N1*N2*ρ*(1+β)*norm_x
    @inbounds @simd for i in eachindex(x, g)
        g[i] += cg*d*x[i]
    end

    return cf*(f - N1*N2*ρ*norm_x)
end

function call(a::Real,
    R::HomogenEdgePreserving{:v2,T},
    x::AbstractArray{T,2}) where {T}

    τ = param(R)
    N1, N2 = size(x)
    ρ = sqrt(2/(N1*N2))*τ 
    α = 2*ρ
    β = 1
    norm_x2 = vnorm2(x)
    norm_x = sqrt(norm_x2)
    f = Float64(0)
    @inbounds for j in 1:N2
        jp = min(j+1, N2)
        @simd for i in 1:N1
            ip = min(i+1, N1)
            # finite differences on average over a 2x2 pixels' cell
            Dx2 = 1/2*(x[i,jp] - x[ip,j])^2 + 1/2*(x[ip,jp] - x[i,j])^2 + 
                   1/6*(x[i,j] - x[ip,j] - x[i,jp] + x[ip,jp])^2
            f += sqrt(Dx2 + ρ^2*norm_x2)
        end
    end

    return a*α*norm_x^β*(f - N1*N2*ρ*norm_x)
end

degree(::HomogenEdgePreserving) = 1.0

homogenedgepreserving(e::T, V::Symbol = :v2) where {T} = 
                     HomogenRegul(HomogenEdgePreserving{V,T}(e))

