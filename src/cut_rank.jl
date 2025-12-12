"""
Cut-rank oracle ρ(A): GF(2) rank of the biadjacency matrix between A and V\\A.

We expose a caching implementation for small-n exact algorithms.
"""

"""
    CutRankOracle(g::BitAdjGraph)

Creates a cut-rank oracle with memoization.
For `n ≤ 64` we memoize results using a `UInt64` subset mask key.
For `n > 64` we still support cut-rank queries (via bitpacked multiword masks),
but do not memoize by default.
"""
mutable struct CutRankOracle
    g::BitAdjGraph
    cache_u64::Union{Nothing, Dict{UInt64, Int}}
end

CutRankOracle(g::BitAdjGraph) = CutRankOracle(g, nvertices(g) <= 64 ? Dict{UInt64, Int}() : nothing)

@inline function _mask_u64_from_vertices(vertices::AbstractVector{Int}, n::Int)::UInt64
    mask = UInt64(0)
    @inbounds for v in vertices
        (1 <= v <= n) || throw(ArgumentError("vertex out of range: $v (expected 1..$n)"))
        mask |= (UInt64(1) << (v - 1))
    end
    return mask
end

@inline function _mask_words_from_vertices(vertices::AbstractVector{Int}, n::Int)::Vector{UInt64}
    nwords = (n + 63) >>> 6
    mask = zeros(UInt64, nwords)
    @inbounds for v in vertices
        (1 <= v <= n) || throw(ArgumentError("vertex out of range: $v (expected 1..$n)"))
        _setbit!(mask, v)
    end
    return mask
end

@inline function _mask_u64_from_bits(bits::AbstractVector{Bool}, n::Int)::UInt64
    length(bits) == n || throw(ArgumentError("mask length mismatch: expected $n, got $(length(bits))"))
    mask = UInt64(0)
    @inbounds for i in 1:n
        if bits[i]
            mask |= (UInt64(1) << (i - 1))
        end
    end
    return mask
end

@inline function _mask_words_from_bits(bits::AbstractVector{Bool}, n::Int)::Vector{UInt64}
    length(bits) == n || throw(ArgumentError("mask length mismatch: expected $n, got $(length(bits))"))
    nwords = (n + 63) >>> 6
    mask = zeros(UInt64, nwords)
    @inbounds for i in 1:n
        if bits[i]
            _setbit!(mask, i)
        end
    end
    return mask
end

"""
    cut_rank(oracle::CutRankOracle, A::UInt64) -> Int

Compute ρ(A) where bit i (1-based) of `A` indicates vertex i is in subset A.
"""
function cut_rank(oracle::CutRankOracle, A::UInt64)::Int
    n = nvertices(oracle.g)
    n <= 64 || throw(ArgumentError(
        "cut_rank(::UInt64) requires n ≤ 64 (got n=$n). " *
        "For n > 64 pass a multiword mask (Vector{UInt64}) or a BitVector/Vector{Bool}."
    ))

    fullmask = n == 64 ? typemax(UInt64) : (UInt64(1) << n) - 1
    A &= fullmask
    key = min(A, (~A) & fullmask) # symmetry: ρ(A)=ρ(V\\A)
    if oracle.cache_u64 !== nothing && haskey(oracle.cache_u64, key)
        return oracle.cache_u64[key]
    end

    B = (~A) & fullmask
    # Build biadjacency rows for vertices in A; columns correspond to vertices in B.
    # For n≤64 we can keep each row as a single-word bitset over V and rely on gf2_rank
    # with ncols=n and mask off non-B columns by zeroing them.
    rows = Vector{Vector{UInt64}}()
    for v in 1:n
        if ((A >>> (v - 1)) & 0x1) == 1
            # neighbors of v into B side
            @inbounds neigh = oracle.g.adj[v][1]  # n≤64 => nwords==1
            push!(rows, UInt64[(neigh & B)])
        end
    end

    # Rank over only B-columns is same as rank over full n columns with non-B columns zeroed.
    r = gf2_rank(rows, n)
    if oracle.cache_u64 !== nothing
        oracle.cache_u64[key] = r
    end
    return r
end

"""
    cut_rank(oracle::CutRankOracle, bits::AbstractVector{Bool}) -> Int

Compute ρ(A) from a boolean indicator vector of length `nvertices(oracle.g)`.

- For `n ≤ 64` this uses the memoized `UInt64` fast path.
- For `n > 64` this packs into a multiword mask and uses the generic path.
"""
function cut_rank(oracle::CutRankOracle, bits::AbstractVector{Bool})::Int
    n = nvertices(oracle.g)
    if n <= 64
        return cut_rank(oracle, _mask_u64_from_bits(bits, n))
    end
    return cut_rank(oracle, _mask_words_from_bits(bits, n))
end

"""
    cut_rank(oracle::CutRankOracle, vertices::AbstractVector{Int}) -> Int

Compute ρ(A) where `vertices` lists the vertices in subset A.

- For `n ≤ 64` this uses the memoized `UInt64` fast path.
- For `n > 64` this packs into a multiword mask and uses the generic path.
"""
function cut_rank(oracle::CutRankOracle, vertices::AbstractVector{Int})::Int
    n = nvertices(oracle.g)
    if n <= 64
        return cut_rank(oracle, _mask_u64_from_vertices(vertices, n))
    end
    return cut_rank(oracle, _mask_words_from_vertices(vertices, n))
end

"""
    cut_rank(oracle::CutRankOracle, Awords::Vector{UInt64}) -> Int

Compute ρ(A) where `Awords` is a bitpacked subset mask over `1..nvertices(oracle.g)`.

This supports arbitrary `n` (including `n > 64`). For `n ≤ 64`, prefer calling
`cut_rank(oracle, ::UInt64)` for the memoized fast path.
"""
function cut_rank(oracle::CutRankOracle, Awords::Vector{UInt64})::Int
    n = nvertices(oracle.g)
    nwords = (n + 63) >>> 6
    length(Awords) == nwords || throw(ArgumentError("mask word length mismatch: expected $nwords, got $(length(Awords))"))

    # full mask over 1..n (mask off unused bits in last word)
    full = Vector{UInt64}(undef, nwords)
    fill!(full, typemax(UInt64))
    extra = nwords * 64 - n
    if extra != 0
        full[end] = typemax(UInt64) >>> extra
    end

    # B = (~A) & full
    B = Vector{UInt64}(undef, nwords)
    @inbounds for i in 1:nwords
        B[i] = (~Awords[i]) & full[i]
    end

    # Build biadjacency rows for vertices in A; columns correspond to vertices in B.
    rows = Vector{Vector{UInt64}}()
    @inbounds for wi in 1:nwords
        x = Awords[wi]
        while x != 0
            b = trailing_zeros(x) # 0..63
            v = (wi - 1) * 64 + b + 1
            v > n && break
            neigh = oracle.g.adj[v]
            row = Vector{UInt64}(undef, nwords)
            @inbounds for j in 1:nwords
                row[j] = neigh[j] & B[j]
            end
            push!(rows, row)
            x &= x - 1
        end
    end

    return gf2_rank(rows, n)
end
