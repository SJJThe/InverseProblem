#
# InverseProblem.jl --
#
# Tools for solving inverse problems
#
#------------------------------------------------------------------------------
#

module InverseProblem

export
    alternated_scaled_solve!,
    alternated_solve!,
    call,
    call!,
    compute_FDMCPSURE,
    edgepreserving,
    HomogenRegul,
    Lkl,
    newton_raphson_solve!,
    norml1,
    norml2,
    powellnewuoa,
    quadraticsmoothness,
    quasinewton,
    Regul,
    Regularization,
    Solver,
    solve,
    solve!,
    SubProblem,
    test_tol

import Base: *
using LazyAlgebra
import LazyAlgebra: input_size, output_size
using OptimPackNextGen
import OptimPackNextGen.Brent
import OptimPackNextGen.Powell.Newuoa
import OptimPackNextGen.Powell.Bobyqa

#TODO: include in doc refernces to other doc

include("types.jl")
include("regularization.jl")
include("solvers.jl")

end # module
