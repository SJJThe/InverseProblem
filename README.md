# InverseProblem

`InverseProblem` is a [Julia](https://julialang.org/) package defining tools
used in the general inverse problems framework. The general problem to solve is
to estimate an array `x` representing the variable $\mathbf{x}$ by solving
```math
\hat{\mathbf{x}} = \arg\min_{\mathbf{x} \in \mathbb{X}} \mathcal{G}(\mathbf{d},\mathbf{x})
```
where $\hat{\mathbf{x}}$ is an estimator of $\mathbf{x}$, $\mathbb{X}$ is the
set of feasible solutions, $\mathbf{d}$ represents some data to fit and
$\mathcal{G}$ is a loss function to minimize.

Currently, the package assumes that the data can be expressed as
```math
\mathbf{d} = \mathbf{A}(\mathbf{x}) + \mathbf{n}
```
with $\mathbf{n}$ accounting for noises in the data and where $\mathbf{A}$ is a
`Mapping` structure of the [LazyAlgebra](https://github.com/emmt/LazyAlgebra.jl)
package.



## Defining elements of a loss function

### `Cost` type

The type `Cost` defined in this package inherits all properties of the `Mapping`
type of the package `LazyAlgebra`. For every structure `C` inheriting the type
`Cost`, it is necessary to define two functions:
 - `call` which will be used to apply `C` to an array `x`, with an optional
   multiplication to a scalar `a`. These following uses of `C` are similar:
   ```julia
   # Computes a*C(x)
   a*call(C, x)
   call(a, C, x)
   C(a, x)
   ```
 - `call!` which will apply `C` to an array `x` and store the gradient in a
   given array `g`. The use of this function is as follows:
   ```julia
   call!(C, x, g)
   C(x, g)
   ```
   As for `call`, an optional scalar can be given. There is also an optional
   boolean keyword `incr` which when put to `true` will add to `g` the gradient
   instead of erasing the values of `g` to store the gradient.


### Compute likelihood

Using the `InverseProblem` package, it is possible to define a likelihood
structure `Lkl`, of type `Cost`, composed of a `Mapping` operator `A`, a data array `d` and
optionally an array `w` (full of ones by default) representing the precision of
the data:
```julia
L = Lkl(A, d, w)
```
As for any other structure of type `Cost`, `L` can be applied to an array `x` as
a `Mapping` operator or with the functions `call` and `call!`:
```julia
# computes likelihood in x
L(x)
call(L, x)
# computes likelihood in x and stores the gradient in g
L(x, g)
call!(L, x, g)
```
which computes
```math
(\mathbf{d} - \mathbf{A}\,\mathbf{x})^T\,\mathbf{W}\,(\mathbf{d} - \mathbf{A}\,\mathbf{x})
```
with $\mathbf{W}$ a diagonal matrix of diagonal elements the ones stored
in `w`.


### Compute regularization functions

A type `Regularization` is defined, inheriting the properties of the type
`Cost`, to group different definition of regularization. From this type, a
`Regul` structure can be used to define a regularization function. To do so, the
user must create a custom structure which inherit from the `Regularization` type
and define the two functions `call` and `call!` associated:
```julia
# Custom structure
struct CustomReg <:Regularization end

# Definition of call! and call
function call!(a::Real, ::CustomReg, x::AbstractArray{T,N}, g::AbstractArray{T,N};
               incr::Bool = false) where {T,N}
    ...
end

function call(a::Real, ::CustomReg, x::AbstractArray{T,N}) where {T,N}
    ...
end
```
These two functions are usually used when solving iteratively an optimization
problem for the computation of the loss function. If the user wants to solve
analytically the problem by directly inverting the Normal equations, it is
possible to define a function yielding the gradient operator:
```julia
function get_grad_op(a::Real, ::CustomReg)
    ...
end
```
The regularization can then be defined as a function:
```julia
mu = 10^3 # hyper-parameter
f = CustomReg() # custom regularization as a structure
direct_inversion = false # or true if CustomReg can be directly inverted

reg() = Regul(mu, f, direct_inversion)
```

### Example of custom regularization function

As for certain optimization method the use of particular regularization
function is needed, it is possible to create its own instance of type
`Regularization`. An example is given in the package for homogeneous
regularization, that is regularization $\mathcal{R}$ that obeys the following property:
```math
\forall \alpha, \mathcal{R}(\alpha\,\mathbf{x}) = \alpha^r\,\mathcal{R}(\mathbf{x})
```
with $r$ the degree of the homogeneous regularization.

As it requires the knowledge of the degree, it is possible to create a new
structure:
```julia
struct HomogenRegul{T} <: Regularization
    Reg::Regul{T}
    deg::T
end
HomogenRegul(mu::Real, f, inv::Bool, deg::Real) = HomogenRegul(Regul(mu, f, inv), deg)

# define the function call! and call, generic for every type Cost
call!(a, R::HomogenRegul, x, g; kwds...) = call!(a, R.Reg, x, g; kwds...)
call(a, R::HomogenRegul, x) = call(a, R.Reg, x)
```
From then, the definition of a custom regularization is as it is described above
except when creating the function `reg()`:
```julia
# function to retrieve the degree of regularization
degree(::CustomReg) = ...
# definition of the regularization function
reg() = HomogenRegul(mu, f, direct_inversion, degree(::CustomReg))
```

Several commonly used regularization are already defined in the file
`regularization.jl`:
 - $\ell 1$ norm;
 - $\ell 2$ norm;
 - Tikhonov;
 - edge-preserving;
 - an homogeneous version of edge-preserving;


### Storing loss function elements

The package provides also an `InvProblem` structure which can be used to store
a loss function in the form of a sum of likelihood and regularization. To define
such loss function, it is possible to call:
```julia
G = InvProblem(A, d, w, R)
```
with the arguments of the structure defined as above. As it is of type `Cost`,
`G` can be used as the other structures described here (either as an operator or
by using `call!` and `call`).



## Using different optimization strategies to solve the minimization problem

...
