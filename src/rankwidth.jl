"""
Exact rank-width (small-n) via subset DP.

This implements the DP described in the project plan:

    dp[∅] = 0
    dp[{v}] = ρ({v})
    dp[S] = min_{A ⊂ S, A≠∅, A≠S} max(dp[A], dp[S\\A])

where ρ(S) is the cut-rank over GF(2) of (S, V\\S).

NOTE: This DP yields an optimal recursive bipartitioning of the vertex set.
The induced tree is binary; it can be converted to a subcubic (ternary) rank
decomposition later (in `rank_decomposition.jl`) if needed.
"""

struct ExactRankWidthResult
    rankwidth::Int
    choice::Dict{UInt64, UInt64}  # choice[S] = A (best split) for |S|≥2
    oracle::CutRankOracle
end

"""
Result of exact linear rank-width (small-n) DP.

- `linear_rankwidth`: the optimal maximum cut-rank over prefixes of the returned order
- `order`: a permutation of 1..n achieving that value
"""
struct ExactLinearRankWidthResult
    linear_rankwidth::Int
    order::Vector{Int}
    choice_last::Dict{UInt64, Int}  # last vertex chosen for subset S
    oracle::CutRankOracle
end

"""
    exact_rankwidth(g::BitAdjGraph; max_n::Int=20) -> ExactRankWidthResult

Compute exact rank-width for `n ≤ 64` using subset DP. For safety we also cap
`n ≤ max_n` (since complexity is exponential).
"""
function exact_rankwidth(g::BitAdjGraph; max_n::Int = 20)::ExactRankWidthResult
    n = nvertices(g)
    n <= 64 || throw(ArgumentError("exact_rankwidth currently supports n ≤ 64 (got n=$n)"))
    n <= max_n || throw(ArgumentError("n=$n exceeds max_n=$max_n for exact DP"))

    full = n == 64 ? typemax(UInt64) : (UInt64(1) << n) - 1
    oracle = CutRankOracle(g)

    dp = Dict{UInt64, Int}()
    choice = Dict{UInt64, UInt64}()

    dp[UInt64(0)] = 0
    for v in 1:n
        S = UInt64(1) << (v - 1)
        dp[S] = cut_rank(oracle, S)
    end

    # Iterate subsets by increasing popcount
    for k in 2:n
        for S in _subsets_of_size(n, k)
            best = typemax(Int)
            bestA = UInt64(0)
            rS = cut_rank(oracle, S)
            # iterate nonempty proper A ⊂ S, but only half to avoid symmetric duplicates
            A = (S - 1) & S
            while A != 0
                B = S ⊻ A
                # enforce canonical A <= B to halve work
                if A <= B
                    val = max(rS, dp[A], dp[B])
                    if val < best
                        best = val
                        bestA = A
                    end
                end
                A = (A - 1) & S
            end
            dp[S] = best
            choice[S] = bestA
        end
    end

    return ExactRankWidthResult(dp[full], choice, oracle)
end

"""
    exact_linear_rankwidth(g::BitAdjGraph; max_n::Int=20, last_vertex::Union{Nothing, Int}=nothing) -> ExactLinearRankWidthResult

Compute exact **linear rank-width** for `n ≤ 64` using subset DP in `O(n 2^n)`.

Linear rank-width is the minimum, over all vertex orderings π, of:

    max_{i=1..n-1} ρ({π₁, …, πᵢ})

where ρ(A) is the cut-rank over GF(2) of the cut (A, V\\A).

If `last_vertex` is provided, the ordering is constrained to end with that vertex.
"""
function exact_linear_rankwidth(g::BitAdjGraph; max_n::Int = 20, last_vertex::Union{Nothing, Int} = nothing)::ExactLinearRankWidthResult
    n = nvertices(g)
    n <= 64 || throw(ArgumentError("exact_linear_rankwidth currently supports n ≤ 64 (got n=$n)"))
    n <= max_n || throw(ArgumentError("n=$n exceeds max_n=$max_n for exact DP"))

    full = n == 64 ? typemax(UInt64) : (UInt64(1) << n) - 1
    oracle = CutRankOracle(g)

    dp = Dict{UInt64, Int}()
    choice_last = Dict{UInt64, Int}()
    dp[UInt64(0)] = 0

    for k in 1:n
        for S in _subsets_of_size(n, k)
            # If last_vertex is constrained, it must only appear at the very end (step n).
            # So if k < n and S contains last_vertex, this S is invalid as a prefix.
            if last_vertex !== nothing && k < n
                if ((S >>> (last_vertex - 1)) & 0x1) == 1
                    continue
                end
            end

            rS = cut_rank(oracle, S)
            best = typemax(Int)
            bestv = 1
            # pick the last vertex added to obtain S
            for v in 1:n
                ((S >>> (v - 1)) & 0x1) == 0 && continue
                
                # If constrained, at step n, v must be last_vertex
                if last_vertex !== nothing && k == n && v != last_vertex
                    continue
                end

                prev = S & ~(UInt64(1) << (v - 1))
                # dp[prev] might be missing if prev was skipped (should not happen if logic is correct)
                # If prev was skipped, it means prev was invalid, so S via v is invalid.
                if !haskey(dp, prev)
                    continue
                end
                
                val = max(dp[prev], rS)
                if val < best
                    best = val
                    bestv = v
                end
            end
            if best != typemax(Int)
                dp[S] = best
                choice_last[S] = bestv
            end
        end
    end

    # reconstruct order from last-choices
    order_rev = Int[]
    S = full
    while S != 0
        v = choice_last[S]
        push!(order_rev, v)
        S &= ~(UInt64(1) << (v - 1))
    end
    order = reverse(order_rev)
    return ExactLinearRankWidthResult(dp[full], order, choice_last, oracle)
end

# Generate all n-bit subsets with popcount == k (UInt64 mask).
# n is small (guarded by max_n), so simple backtracking is fine and robust.
function _subsets_of_size(n::Int, k::Int)::Vector{UInt64}
    out = UInt64[]
    _collect_subsets!(out, n, k, 1, UInt64(0))
    return out
end

function _collect_subsets!(out::Vector{UInt64}, n::Int, k::Int, start::Int, cur::UInt64)
    k == 0 && (push!(out, cur); return)
    # choose remaining k positions from [start..n]
    # prune when not enough elements remain
    for i in start:(n - k + 1)
        _collect_subsets!(out, n, k - 1, i + 1, cur | (UInt64(1) << (i - 1)))
    end
    return
end
