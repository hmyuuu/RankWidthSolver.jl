module RankWidthSolver

using Graphs
using AbstractTrees
using OMEinsumContractionOrders

# Public API (keep small and conventional; users can still access internals)
export gf2_rank
export BitAdjGraph, from_simple_graph, nvertices, neighbors_mask
export CutRankOracle, cut_rank
export exact_rankwidth, exact_linear_rankwidth
export DecompNode, build_decomposition, leaves_mask
export elimination_order
export ExactRankWidth, ApproxRankWidth, ExactLinearRankWidth
export caterpillar_to_path

include("gf2.jl")
include("graphs.jl")
include("cut_rank.jl")
include("rankwidth.jl")
include("rank_decomposition.jl")
include("contraction_order.jl")
include("omeinsum_optimizer.jl")

end # module RankWidthSolver

