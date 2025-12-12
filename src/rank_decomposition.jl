"""
Reconstruction of a (binary) decomposition tree from the subset-DP `choice` table.

The DP in `rankwidth.jl` produces a recursive bipartition tree. This is a valid
rooted decomposition for our purposes (and can be transformed into a subcubic
rank decomposition if needed later).
"""

mutable struct DecompNode
    mask::UInt64
    parent::Union{DecompNode, Nothing}
    children::Vector{DecompNode}
end

DecompNode(mask::UInt64) = DecompNode(mask, nothing, DecompNode[])

AbstractTrees.children(n::DecompNode) = n.children
AbstractTrees.ParentLinks(::Type{<:DecompNode}) = StoredParents()
AbstractTrees.parent(n::DecompNode) = n.parent
AbstractTrees.nodevalue(n::DecompNode) = n.mask

leaves_mask(n::DecompNode) = n.mask

"""
    build_decomposition(fullmask::UInt64, choice::Dict{UInt64,UInt64}) -> DecompNode

Reconstruct a binary decomposition tree:
- if `choice[S] = A`, then children are `A` and `S\\A`.
"""
function build_decomposition(fullmask::UInt64, choice::Dict{UInt64, UInt64})::DecompNode
    root = DecompNode(fullmask)
    _build!(root, choice)
    return root
end

function _build!(node::DecompNode, choice::Dict{UInt64, UInt64})
    S = node.mask
    # leaf if singleton or empty or no recorded choice
    if count_ones(S) <= 1 || !haskey(choice, S)
        return nothing
    end
    A = choice[S]
    B = S ⊻ A
    c1 = DecompNode(A); c1.parent = node
    c2 = DecompNode(B); c2.parent = node
    node.children = [c1, c2]
    _build!(c1, choice)
    _build!(c2, choice)
    return nothing
end
