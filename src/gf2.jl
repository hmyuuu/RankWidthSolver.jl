"""
GF(2) linear algebra primitives used by rank-width cut-rank computations.

We implement an allocation-light rank computation for matrices over GF(2),
represented as bitpacked rows (`UInt64` words).
"""

"""
    gf2_rank(rows::Vector{Vector{UInt64}}, ncols::Int) -> Int

Compute the rank over GF(2) of a binary matrix given by `rows`.

- `rows[i]` is a bitpacked row vector of length `ncols` bits stored in `UInt64` words.
- Bits are interpreted little-endian within each word: bit `j` of the row is stored in
  word `w = (j-1)>>>6 + 1` at bit position `(j-1) & 63`.

This function performs Gaussian elimination using XOR row operations.
It mutates a local copy of the pivot rows, but does not mutate `rows`.
"""
function gf2_rank(rows::Vector{Vector{UInt64}}, ncols::Int)::Int
    isempty(rows) && return 0
    nwords = (ncols + 63) >>> 6
    pivots = Vector{Union{Nothing, Vector{UInt64}}}(nothing, ncols) # pivot by leading bit (1..ncols)

    rank = 0
    for r in rows
        row = copy(r)
        @inbounds while true
            lead = _leading_one(row, ncols)
            lead == 0 && break
            p = pivots[lead]
            if p === nothing
                pivots[lead] = row
                rank += 1
                break
            else
                _xor!(row, p, nwords)
            end
        end
    end
    return rank
end
