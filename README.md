# RankWidthSolver

RankWidthSolver is a Julia package that computes **rank-width** (via the **cut-rank function** over GF(2)) and uses it to build contraction orders for tensor networks, integrating as an optimizer for `OMEinsumContractionOrders.jl`.

References: [Rank-width (Wikipedia)](https://en.wikipedia.org/wiki/Rank-width), [Oum survey (arXiv:1601.03800)](https://arxiv.org/pdf/1601.03800)

## Usage

### OMEinsum / OMEinsumContractionOrders

```julia
using OMEinsum, OMEinsumContractionOrders
using RankWidthSolver

code = ein"ij,jk,kl->il"
size_dict = uniformsize(code, 2)

# exact (small-n) rank-width DP optimizer
opt = RankWidthSolver.OMEinsumIntegration.ExactRankWidth(max_n=20)
optcode = optimize_code(code, size_dict, opt)

@show contraction_complexity(optcode, size_dict)
```

### Complete graph benchmark

Run:

```bash
julia --project=. examples/complete_graph_benchmark.jl
```

For complete graphs `Kₙ` with `n≥2`, rank-width is `1` because every non-trivial cut yields an all-ones biadjacency matrix of GF(2)-rank 1.

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

