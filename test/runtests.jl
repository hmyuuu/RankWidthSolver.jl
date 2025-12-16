using Test
using RankWidthSolver
using Graphs
using OMEinsum
using OMEinsumContractionOrders

@testset "RankWidthSolver.jl" begin
    @testset "GF2 rank" begin
        rows = [UInt64[0x3], UInt64[0x1], UInt64[0x2]]
        @test gf2_rank(rows, 2) == 2
    end

    @testset "cut-rank symmetry" begin
        g = path_graph(6)
        bg = from_simple_graph(g)
        oracle = CutRankOracle(bg)
        full = (UInt64(1) << 6) - 1
        A = UInt64(0b001011) # {1,2,4}
        @test cut_rank(oracle, A) ==
              cut_rank(oracle, (~A) & full)
    end

    @testset "exact rankwidth on complete graphs" begin
        for n in 2:8
            g = complete_graph(n)
            bg = from_simple_graph(g)
            res = exact_rankwidth(bg; max_n=20)
            @test res.rankwidth == 1
        end
    end

    @testset "OMEinsum optimizer integration" begin
        code = ein"ij,jk,kl->il"
        sd = uniformsize(code, 2)
        opt = ExactRankWidth(max_n=20)
        optcode = optimize_code(code, sd, opt)
        A = rand(2,2); B = rand(2,2); C = rand(2,2)
        @test code(A,B,C) ≈ optcode(A,B,C)
    end

    @testset "ExactLinearRankWidth optimizer integration" begin
        code = ein"ij,jk,kl->il"
        sd = uniformsize(code, 2)
        opt = ExactLinearRankWidth(max_n=20)
        optcode = optimize_code(code, sd, opt)
        A = rand(2,2); B = rand(2,2); C = rand(2,2)
        @test code(A,B,C) ≈ optcode(A,B,C)
    end

    @testset "n > 64 support (word masks)" begin
        n = 130
        g = path_graph(n)
        bg = from_simple_graph(g)
        oracle = CutRankOracle(bg)

        nwords = (n + 63) >>> 6
        Awords = zeros(UInt64, nwords)
        for v in 1:2:n
            RankWidthSolver._setbit!(Awords, v)
        end
        rA = cut_rank(oracle, Awords)

        # symmetry: ρ(A) = ρ(V\A)
        full = Vector{UInt64}(undef, nwords)
        fill!(full, typemax(UInt64))
        extra = nwords * 64 - n
        if extra != 0
            full[end] = typemax(UInt64) >>> extra
        end
        Bwords = Vector{UInt64}(undef, nwords)
        @inbounds for i in 1:nwords
            Bwords[i] = (~Awords[i]) & full[i]
        end
        @test rA == cut_rank(oracle, Bwords)

        # greedy linear order should produce a permutation of 1..n without throwing
        order = RankWidthSolver._greedy_linear_order(bg)
        @test length(order) == n
        @test sort(order) == collect(1:n)

        # approx recursive parts should cover all vertices exactly once
        parts = RankWidthSolver._approx_parts(collect(1:n), oracle, 20)
        function _flatten_parts(x)
            x isa Int && return [x]
            out = Int[]
            for y in x
                append!(out, _flatten_parts(y))
            end
            return out
        end
        flat = _flatten_parts(parts)
        @test length(flat) == n
        @test sort(flat) == collect(1:n)
    end

    @testset "Path Structure Enforcement" begin
        # Test 1: Transformation of a balanced tree to a path
        ixs = [[1], [1, 2], [2, 3], [3]]
        iy = Int[]
        
        # Construct leaves using the constructor that works (assuming single int constructor exists or using explicit one)
        # Note: In OMEinsumContractionOrders, leaves are NestedEinsum with empty args and a tensor index.
        # We rely on the fact that existing tests passed, so NestedEinsum{Int}(i) likely works.
        L1 = OMEinsumContractionOrders.NestedEinsum{Int}(1)
        L2 = OMEinsumContractionOrders.NestedEinsum{Int}(2)
        L3 = OMEinsumContractionOrders.NestedEinsum{Int}(3)
        L4 = OMEinsumContractionOrders.NestedEinsum{Int}(4)

        # Helper to validate path structure (caterpillar: args[2] is always a leaf)
        function validate_path(code)
            if !(code isa OMEinsumContractionOrders.NestedEinsum) || isempty(code.args)
                return true
            end
            @test length(code.args) == 2
            left, right = code.args[1], code.args[2]
            # Right child MUST be a leaf (empty args)
            if right isa OMEinsumContractionOrders.NestedEinsum
                @test isempty(right.args)
            end
            validate_path(left)
        end
        
        # Test: ExactLinearRankWidth produces path structure
        code = ein"ij,jk,kl,lm->im"
        sd = uniformsize(code, 2)
        opt = ExactLinearRankWidth(max_n=20)
        optcode = optimize_code(code, sd, opt)
        validate_path(optcode)
        
        # Test: Correctness of contraction
        A = rand(2,2); B = rand(2,2); C = rand(2,2); D = rand(2,2)
        res_orig = code(A,B,C,D)
        res_opt = optcode(A,B,C,D)
        @test res_orig ≈ res_opt
    end
end # RankWidthSolver.jl tests
