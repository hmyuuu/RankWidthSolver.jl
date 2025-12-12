"""
Convert a decomposition tree into a simple elimination / contraction order.

For now, we output an elimination order as a vector of vertex sets, similar in spirit
to TreeWidthSolver's `EliminationOrder`: each inner vector is a group eliminated at once.
"""

"""
    elimination_order(tree::DecompNode) -> Vector{Vector{Int}}

Postorder traversal; each leaf contributes one vertex id. Because our decomposition
tree leaves correspond to singleton masks, we return a simple sequence of singletons.
"""
function elimination_order(tree::DecompNode)::Vector{Vector{Int}}
    order = Vector{Vector{Int}}()
    for node in PostOrderDFS(tree)
        if isempty(children(node))
            v = _singleton_vertex(node.mask)
            push!(order, [v])
        end
    end
    return order
end

function _singleton_vertex(mask::UInt64)::Int
    @assert count_ones(mask) == 1
    return trailing_zeros(mask) + 1
end
