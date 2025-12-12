using RankWidthSolver
using Graphs

println("Complete graph rank-width benchmark (exact DP, small n)")
for n in (2, 3, 4, 5, 6, 7, 8, 10)
    g = complete_graph(n)
    bg = from_simple_graph(g)
    res = exact_rankwidth(bg; max_n=20)
    println("K_$n: rankwidth = ", res.rankwidth, " (expected 1)")
end

