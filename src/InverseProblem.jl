#
# InverseProblem.jl
#
# Tools for solving inverse problems.
#
#------------------------------------------------------------------------------
#



"""
    InverseProblem

This package aims at developing tools that are used in the general inverse
problem framework.

"""
module InverseProblem

export
    BilinearInverseProblem,
    BilinearProblem,
    brentmin,
    build_Vandermonde_matrix,
    call,
    call!,
    Cost,
    degree,
    edgepreserving,
    fit_polynomial_law,
    get_grad_op,
    homogenedgepreserving,
    HomogenRegul,
    InvProblem,
    Lkl,
    multiplier,
    norml1,
    norml2,
    PolynLaw,
    PolynMdl,
    powellbobyqa,
    powellnewuoa,
    tikhonov,
    quasinewton,
    Regul,
    Regularization,
    Solver,
    solve,
    solve!,
    SumRegul,
    test_tol,
    use_direct_inversion,
    WeightedTikhonov

import Base: *
using LazyAlgebra
import LazyAlgebra: input_size, output_size
using LinearAlgebra
using LinearInterpolators
using OptimPackNextGen
import OptimPackNextGen.Brent
import OptimPackNextGen.Powell.Newuoa
import OptimPackNextGen.Powell.Bobyqa
using Statistics
using TwoDimensional


include("types.jl")
include("regularization.jl")
include("solvers.jl")
include("fit_pol_tools.jl")

end # module
