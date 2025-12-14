"""
    _approx_parts(vertices::Vector{Int}, oracle::CutRankOracle, max_group_size::Int)

Heuristic recursive bipartition of `vertices` based on cut-rank.
Returns a nested vector structure (compatible with OMEinsum's format).
"""
function _approx_parts(vertices::Vector{Int}, oracle::CutRankOracle, max_group_size::Int)
    n = length(vertices)
    n <= max_group_size && return [vertices]
    n == 1 && return [vertices]
    n == 2 && return [[vertices[1]], [vertices[2]]]

    ng = nvertices(oracle.g)
    if ng <= 64
        Smask = UInt64(0)
        for v in vertices
            Smask |= (UInt64(1) << (v - 1))
        end

        # Greedily build A until roughly half of S
        A = UInt64(1) << (vertices[1] - 1)
        remaining = Set(vertices[2:end])
        target = n >>> 1

        while count_ones(A) < target && !isempty(remaining)
            bestv = first(remaining)
            bestscore = typemax(Int)
            for v in remaining
                A2 = A | (UInt64(1) << (v - 1))
                B2 = Smask ⊻ A2
                score = max(cut_rank(oracle, A2), cut_rank(oracle, B2))
                if score < bestscore
                    bestscore = score
                    bestv = v
                end
            end
            delete!(remaining, bestv)
            A |= (UInt64(1) << (bestv - 1))
        end

        B = Smask ⊻ A
        Averts = Int[]
        Bverts = Int[]
        for v in vertices
            if ((A >>> (v - 1)) & 0x1) == 1
                push!(Averts, v)
            else
                push!(Bverts, v)
            end
        end

        return [_approx_parts(Averts, oracle, max_group_size),
                _approx_parts(Bverts, oracle, max_group_size)]
    end

    # n > 64 path: use multiword subset masks
    nwords = (ng + 63) >>> 6
    Smask = zeros(UInt64, nwords)
    for v in vertices
        _setbit!(Smask, v)
    end

    A = zeros(UInt64, nwords)
    _setbit!(A, vertices[1])
    remaining = Set(vertices[2:end])
    target = n >>> 1
    acount = 1

    scratchA = similar(A)
    scratchB = similar(A)

    while acount < target && !isempty(remaining)
        bestv = first(remaining)
        bestscore = typemax(Int)
        for v in remaining
            copyto!(scratchA, A)
            _setbit!(scratchA, v)
            @inbounds for i in 1:nwords
                scratchB[i] = Smask[i] ⊻ scratchA[i]
            end
            score = max(cut_rank(oracle, scratchA), cut_rank(oracle, scratchB))
            if score < bestscore
                bestscore = score
                bestv = v
            end
        end
        delete!(remaining, bestv)
        _setbit!(A, bestv)
        acount += 1
    end

    B = similar(A)
    @inbounds for i in 1:nwords
        B[i] = Smask[i] ⊻ A[i]
    end

    Averts = Int[]
    Bverts = Int[]
    for v in vertices
        w = (v - 1) >>> 6 + 1
        b = (v - 1) & 63
        if ((A[w] >>> b) & 0x1) == 1
            push!(Averts, v)
        else
            push!(Bverts, v)
        end
    end

    return [_approx_parts(Averts, oracle, max_group_size),
            _approx_parts(Bverts, oracle, max_group_size)]
end

function _order_to_left_parts(order::Vector{Int})
    length(order) == 1 && return [order[1]]
    return [_order_to_left_parts(order[1:end-1]), [order[end]]]
end

"""
    _greedy_linear_order(g::BitAdjGraph; max_group_size::Int = 40)

Greedy heuristic for linear rank-width.
Builds a prefix by iteratively adding the vertex that minimizes the cut-rank of the prefix.
"""
function _greedy_linear_order(g::BitAdjGraph; max_group_size::Int = 40)
    # Simple greedy: build prefix, each step add vertex minimizing resulting prefix cut-rank.
    n = nvertices(g)
    oracle = CutRankOracle(g)
    remaining = Set(1:n)
    order = Int[]
    if n <= 64
        prefix = UInt64(0)
        while !isempty(remaining)
            bestv = first(remaining)
            best = typemax(Int)
            for v in remaining
                S = prefix | (UInt64(1) << (v - 1))
                score = cut_rank(oracle, S)
                if score < best
                    best = score
                    bestv = v
                end
            end
            push!(order, bestv)
            delete!(remaining, bestv)
            prefix |= (UInt64(1) << (bestv - 1))
        end
        return order
    end

    nwords = (n + 63) >>> 6
    prefix = zeros(UInt64, nwords)
    scratch = similar(prefix)
    while !isempty(remaining)
        bestv = first(remaining)
        best = typemax(Int)
        for v in remaining
            copyto!(scratch, prefix)
            _setbit!(scratch, v)
            score = cut_rank(oracle, scratch)
            if score < best
                best = score
                bestv = v
            end
        end
        push!(order, bestv)
        delete!(remaining, bestv)
        _setbit!(prefix, bestv)
    end
    return order
end
