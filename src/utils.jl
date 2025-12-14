"""
    _setbit!(row::Vector{UInt64}, idx::Int)

Sets the `idx`-th bit (1-based) in a bitpacked `UInt64` vector.
"""
@inline function _setbit!(row::Vector{UInt64}, idx::Int)
    w = (idx - 1) >>> 6 + 1
    b = (idx - 1) & 63
    @inbounds row[w] |= (UInt64(1) << b)
    return nothing
end

"""
    _xor!(a::Vector{UInt64}, b::Vector{UInt64}, nwords::Int)

Computes `a .⊻= b` in-place for the first `nwords` elements.
"""
@inline function _xor!(a::Vector{UInt64}, b::Vector{UInt64}, nwords::Int)
    @inbounds for i in 1:nwords
        a[i] ⊻= b[i]
    end
    return nothing
end

"""
    _leading_one(row::Vector{UInt64}, ncols::Int) -> Int

Finds the index (1-based) of the highest set bit within the first `ncols` bits.
Returns 0 if all bits are zero.
"""
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
