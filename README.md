# RankWidthSolver

RankWidthSolver is a Julia package that computes **rank-width** (via the **cut-rank function** over GF(2)) and uses it to build contraction orders for tensor networks, integrating as an optimizer for `OMEinsumContractionOrders.jl`.

References:  [Oum survey (arXiv:1601.03800)](https://arxiv.org/pdf/1601.03800)

## Usage

### OMEinsum / OMEinsumContractionOrders

```julia
using OMEinsum, OMEinsumContractionOrders
using RankWidthSolver

code = ein"ij,jk,kl->il"
size_dict = uniformsize(code, 2)

# exact (small-n) rank-width DP optimizer
optcode = optimize_code(code, size_dict, ExactRankWidth())

@show contraction_complexity(optcode, size_dict)
```

## Installation

<p>
RankWidthSolver is a &nbsp;
    <a href="https://julialang.org">
        <img src="https://raw.githubusercontent.com/JuliaLang/julia-logo-graphics/master/images/julia.ico" width="16em">
        Julia Language
    </a>
    &nbsp; package. To install RankWidthSolver,
    please <a href="https://docs.julialang.org/en/v1/manual/getting-started/">open
    Julia's interactive session (known as REPL)</a> and press <kbd>]</kbd>
    key in the REPL to use the package mode, then type the following command
</p>

```julia
pkg> add RankWdthSolver
```

