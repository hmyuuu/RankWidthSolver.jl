# OMEinsumContractionOrders.jl integration.
#
# We implement `CodeOptimizer`s that plug into `optimize_code` by extending
# `OMEinsumContractionOrders._optimize_code`.
import OMEinsumContractionOrders:
    CodeOptimizer,
    _optimize_code,
    getixsv,
    getiyv,
    pivot_tree,
    fix_binary_tree,
    recursive_construct_nestedeinsum,
    GreedyMethod

"""
    ExactRankWidth(; max_n=20, sub_optimizer=GreedyMethod())

Exact (small-n) rank-width based contraction optimizer.

This optimizer:
1) builds a *tensor line graph* (tensors as vertices, edges if tensors share an index),
2) computes an exact binary decomposition via subset DP (exponential; guarded by `max_n`),
3) maps that decomposition into the recursive bipartition format used internally by
   OMEinsumContractionOrders to build a `NestedEinsum`.
"""
Base.@kwdef struct ExactRankWidth{SO} <: CodeOptimizer
    max_n::Int = 20
    sub_optimizer::SO = GreedyMethod()
end

function _optimize_code(code, size_dict, optimizer::ExactRankWidth)
    ixs, iy = getixsv(code), getiyv(code)
    ixv = [ixs..., iy]

    lg = _tensor_line_graph(ixv)
    bg = from_simple_graph(lg)
    res = exact_rankwidth(bg; max_n = optimizer.max_n)

    n = nv(lg)
    full = n == 64 ? typemax(UInt64) : (UInt64(1) << n) - 1
    tree = build_decomposition(full, res.choice)
    parts = _tree_to_parts(tree)

    optcode = recursive_construct_nestedeinsum(ixv, empty(iy), parts, size_dict, 0, optimizer.sub_optimizer)
    return fix_binary_tree(pivot_tree(optcode, length(ixs) + 1))
end

"""
    ApproxRankWidth(; max_group_size=20, sub_optimizer=GreedyMethod())

Heuristic (approximate) rank-width based optimizer.

Currently implemented as a greedy recursive bipartition of the tensor line graph,
guided by the cut-rank oracle ρ(·). This is **not** the full Oum 2008 algorithm;
it is a practical fallback that avoids the exponential subset DP.
"""
Base.@kwdef struct ApproxRankWidth{SO} <: CodeOptimizer
    max_group_size::Int = 20
    sub_optimizer::SO = GreedyMethod()
end

function _optimize_code(code, size_dict, optimizer::ApproxRankWidth)
    ixs, iy = getixsv(code), getiyv(code)
    ixv = [ixs..., iy]

    lg = _tensor_line_graph(ixv)
    n = nv(lg)

    bg = from_simple_graph(lg)
    oracle = CutRankOracle(bg)
    vertices = collect(1:n)
    parts = _approx_parts(vertices, oracle, optimizer.max_group_size)

    optcode = recursive_construct_nestedeinsum(ixv, empty(iy), parts, size_dict, 0, optimizer.sub_optimizer)
    return fix_binary_tree(pivot_tree(optcode, length(ixs) + 1))
end

"""
    ExactLinearRankWidth(; max_n=20, max_group_size=40, sub_optimizer=GreedyMethod())

Linear rank-width optimizer (caterpillar/path-like decomposition):

- builds a tensor line graph
- finds a vertex ordering minimizing the maximum prefix cut-rank ρ(prefix)
  - exact DP if `n ≤ max_n`
  - greedy heuristic otherwise
- converts the order into a **left-branching** binary contraction tree (right child is a leaf),
  analog to a path/caterpillar decomposition in treewidth.
"""
Base.@kwdef struct ExactLinearRankWidth{SO} <: CodeOptimizer
    max_n::Int = 20
    max_group_size::Int = 40
    sub_optimizer::SO = GreedyMethod()
end

function _optimize_code(code, size_dict, optimizer::ExactLinearRankWidth)
    ixs, iy = getixsv(code), getiyv(code)
    ixv = [ixs..., iy]

    lg = _tensor_line_graph(ixv)
    n = nv(lg)
    n_inputs = length(ixs)
    
    bg = from_simple_graph(lg)
    # Optimize order over input tensors only (1..n_inputs)
    # The output tensor (n_inputs + 1) will be handled by pivot_tree
    order = if n_inputs <= optimizer.max_n
        n_inputs <= 64 || throw(ArgumentError("ExactLinearRankWidth exact DP currently supports n ≤ 64 (got n=$n_inputs); set max_n < n to use greedy mode"))
        full_order = exact_linear_rankwidth(bg; max_n=optimizer.max_n).order
        # Filter to only include input tensor indices (1..n_inputs)
        filter(v -> v <= n_inputs, full_order)
    else
        full_order = _greedy_linear_order(bg; max_group_size=optimizer.max_group_size)
        filter(v -> v <= n_inputs, full_order)
    end

    parts = _order_to_left_parts(order)
    optcode = recursive_construct_nestedeinsum(ixv, empty(iy), parts, size_dict, 0, optimizer.sub_optimizer)
    # pivot_tree is needed to position the output tensor correctly (at index n_inputs + 1)
    # This ensures the contraction produces the correct output shape
    optcode = pivot_tree(optcode, n_inputs + 1)
    optcode = fix_binary_tree(optcode)
    return caterpillar_to_path(optcode, ixs, iy)
end


function _tensor_line_graph(ixv::AbstractVector{<:AbstractVector})
    n = length(ixv)
    g = SimpleGraph(n)
    buckets = Dict{Any, Vector{Int}}()
    for (v, ix) in enumerate(ixv)
        for l in ix
            push!(get!(buckets, l, Int[]), v)
        end
    end
    for vs in values(buckets)
        m = length(vs)
        for i in 1:m-1, j in i+1:m
            u = vs[i]; v = vs[j]
            u != v && add_edge!(g, u, v)
        end
    end
    return g
end

function _tree_to_parts(node::DecompNode)
    if isempty(children(node))
        return [trailing_zeros(node.mask) + 1]
    end
    cs = children(node)
    @assert length(cs) == 2
    return [_tree_to_parts(cs[1]), _tree_to_parts(cs[2])]
end

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

function _greedy_linear_order(g::BitAdjGraph; max_group_size::Int = 40, last_vertex::Union{Nothing, Int}=nothing)
    # Simple greedy: build prefix, each step add vertex minimizing resulting prefix cut-rank.
    n = nvertices(g)
    oracle = CutRankOracle(g)
    remaining = Set(1:n)
    if last_vertex !== nothing
        delete!(remaining, last_vertex)
    end

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
        if last_vertex !== nothing
            push!(order, last_vertex)
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
    if last_vertex !== nothing
        push!(order, last_vertex)
    end
    return order
end

"""
    caterpillar_to_path(code::OMEinsumContractionOrders.NestedEinsum, ixs, iy)

Transform an arbitrary contraction tree (e.g. balanced) into a strict left-associative path (caterpillar)
where the right child of every node is a leaf tensor.
"""
function caterpillar_to_path(code::OMEinsumContractionOrders.NestedEinsum{L}, ixs, iy) where L
    leaves = _collect_leaves(code)
    if length(leaves) < 2
        return code
    end

    # Extract indices for each leaf
    leaf_inds_list = Vector{Vector{L}}(undef, length(leaves))
    for (i, l) in enumerate(leaves)
        leaf_inds_list[i] = ixs[l.tensorindex]
    end

    n = length(leaves)
    
    # Precompute suffix unions to know which indices are needed later
    # suffix_unions[i] = union of indices in leaves[i...n] AND iy
    suffix_unions = Vector{Vector{L}}(undef, n)
    current_suffix = collect(L, iy)
    for i in n:-1:1
        current_suffix = union(current_suffix, leaf_inds_list[i])
        suffix_unions[i] = current_suffix
    end

    # Start with the first leaf
    current_node = leaves[1]
    current_inds = leaf_inds_list[1]

    for i in 2:n
        next_leaf = leaves[i]
        next_inds = leaf_inds_list[i]
        
        # Determine output indices: (current ∪ next) ∩ (needed later)
        needed = (i < n) ? suffix_unions[i+1] : iy
        all_inds = union(current_inds, next_inds)
        # Use intersect(needed, all_inds) to prefer order of 'needed' (important for final output iy)
        out_inds = intersect(needed, all_inds)
        
        eins = OMEinsumContractionOrders.EinCode([current_inds, next_inds], out_inds)
        
        # Construct new node: Left=current, Right=next_leaf
        # Note: NestedEinsum constructor signature might vary. 
        # Usually: NestedEinsum(args, eins)
        current_node = OMEinsumContractionOrders.NestedEinsum([current_node, next_leaf], eins)
        current_inds = out_inds
    end

    return current_node
end

function _collect_leaves(code::OMEinsumContractionOrders.NestedEinsum{L}) where L
    if isempty(code.args)
        return [code]
    end
    res = OMEinsumContractionOrders.NestedEinsum{L}[]
    for arg in code.args
        if arg isa OMEinsumContractionOrders.NestedEinsum
            append!(res, _collect_leaves(arg))
        else
            # Should not happen in standard OMEinsumContractionOrders usage where leaves are NestedEinsum with empty args
            # But if it does, ignore or handle?
        end
    end
    return res
end
