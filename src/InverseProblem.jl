#
# InverseProblem.jl --
#
# Tools for solving inverse problems.
#
#------------------------------------------------------------------------------
#


#TODO: include in doc references to other doc
#TODO: SURE computation
#TODO: generalize the precision matrix type (eg ASAP)
#TODO: complete README.md
#FIXME: add the call to a sum of :Cost
#FIXME: rename SubProblem into InvProblem

module InverseProblem

export
    brentmin,
    call,
    call!,
    edgepreserving,
    homogenedgepreserving,
    HomogenRegul,
    InvProblem,
    Lkl,
    newton_raphson_solve!,
    norml1,
    norml2,
    powellnewuoa,
    tikhonov,
    quasinewton,
    Regul,
    Regularization,
    Solver,
    solve,
    solve!,
    test_tol

import Base: *
using LazyAlgebra
import LazyAlgebra: input_size, output_size
using OptimPackNextGen
import OptimPackNextGen.Brent
import OptimPackNextGen.Powell.Newuoa
import OptimPackNextGen.Powell.Bobyqa


include("types.jl")
include("regularization.jl")
include("solvers.jl")

end # module
