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

# Find index (1-based) of highest set bit within first ncols bits; 0 if zero.
function _leading_one(row::Vector{UInt64}, ncols::Int)::Int
    nwords = (ncols + 63) >>> 6
    @inbounds for wi in nwords:-1:1
        x = row[wi]
        if x != 0
            bit = 64 - leading_zeros(x)  # 1..64
            idx = (wi - 1) * 64 + bit
            return idx <= ncols ? idx : _leading_one_masked(row, ncols, wi)
        end
    end
    return 0
end

function _leading_one_masked(row::Vector{UInt64}, ncols::Int, wi_start::Int)::Int
    # Handle the top (partial) word: mask off bits above ncols.
    nwords = (ncols + 63) >>> 6
    @assert wi_start == nwords
    extra = nwords * 64 - ncols
    mask = extra == 0 ? typemax(UInt64) : (typemax(UInt64) >>> extra)
    x = row[nwords] & mask
    if x != 0
        bit = 64 - leading_zeros(x)
        return (nwords - 1) * 64 + bit
    end
    @inbounds for wi in (nwords - 1):-1:1
        x2 = row[wi]
        if x2 != 0
            bit2 = 64 - leading_zeros(x2)
            return (wi - 1) * 64 + bit2
        end
    end
    return 0
end

@inline function _xor!(a::Vector{UInt64}, b::Vector{UInt64}, nwords::Int)
    @inbounds for i in 1:nwords
        a[i] ⊻= b[i]
    end
    return nothing
end
