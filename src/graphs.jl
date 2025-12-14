"""
Graph storage for rank-width computations.

We store undirected simple graphs as bitpacked adjacency rows for fast cut-rank queries.
"""

"""
    BitAdjGraph

Undirected simple graph with adjacency stored as bitpacked rows.

- Vertices are 1..n
- `adj[i]` is a bitset of neighbors of i (within 1..n).
"""
struct BitAdjGraph
    n::Int
    adj::Vector{Vector{UInt64}} # length n, each row has nwords words
end

@inline nvertices(g::BitAdjGraph) = g.n

"""
    from_simple_graph(g::Graphs.SimpleGraph) -> BitAdjGraph
"""
function from_simple_graph(g::Graphs.SimpleGraph)
    n = nv(g)
    nwords = (n + 63) >>> 6
    adj = [zeros(UInt64, nwords) for _ in 1:n]
    for e in edges(g)
        u = src(e); v = dst(e)
        _setbit!(adj[u], v)
        _setbit!(adj[v], u)
    end
    return BitAdjGraph(n, adj)
end

"""
    neighbors_mask(g::BitAdjGraph, v::Int) -> Vector{UInt64}

Returns the bitset row (a view/copy) for neighbors of vertex v.
"""
@inline neighbors_mask(g::BitAdjGraph, v::Int) = g.adj[v]
