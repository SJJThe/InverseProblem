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
    call,
    call!,
    Cost,
    degree,
    edgepreserving,
    get_grad_op,
    homogenedgepreserving,
    HomogenRegul,
    InvProblem,
    Lkl,
    multiplier,
    norml1,
    norml2,
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
using OptimPackNextGen
import OptimPackNextGen.Brent
import OptimPackNextGen.Powell.Newuoa
import OptimPackNextGen.Powell.Bobyqa
using Statistics


include("types.jl")
include("regularization.jl")
include("solvers.jl")

end # module
