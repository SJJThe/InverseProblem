#
# fit_tools.jl
#
# Closed-form fit of polynomial laws.
#
#------------------------------------------------------------------------------
#
# This file is part of InverseProblem

"""
"""
function fit_polynomial_law(polynom_caracs::NTuple{2,Int},
    coords::AbstractVector{M},
    data::AbstractVector{T}) where {T,M<:Union{T,Point{T}}}

    H = build_Vandermonde_matrix(polynom_caracs, coords)
    A = H'*H
    b = H'*data

    return ldiv!(cholesky!(Symmetric(A)), b)
end




"""
    build_Vandermonde_matrix(polynom_caracs, coords) -> H

yields the Vandermonde matrix `H` coming from the mean square fit of a
polynomial law of caracteristics the 2-Tuple `polynom_caracs` (dimension,
degree).

see also: [PolynMdl](@ref)

"""
function build_Vandermonde_matrix(polynom_caracs::NTuple{2,Int},
    coords::AbstractVector{M}) where {T,M<:Union{T,Point{T}}}

    H = Matrix{Float64}[]
    for p in coords
        Pmdl = get_mdl(PolynMdl{polynom_caracs[1],T,polynom_caracs[2]}(p))
        v = []
        for i in eachindex(Pmdl)
            v = append!(v, Pmdl[i])
        end
        push!(H, v')
    end

    return vcat(H...)
end



