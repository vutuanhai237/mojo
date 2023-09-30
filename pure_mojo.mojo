from benchmark import Benchmark
from sys.intrinsics import strided_load
from utils.list import VariadicList
from math import div_ceil, min
from memory import memset_zero
from memory.unsafe import DTypePointer
from random import rand, random_float64
from sys.info import simdwidthof
from runtime.llcl import Runtime

fn matmul_vectorized_0(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):
            for nv in range(0, C.cols, nelts):
                C.store[nelts](
                    m, nv, C.load[nelts](m, nv) + A[m, k] * B.load[nelts](k, nv)
                )

            # Handle remaining elements with scalars.
            for n in range(nelts * (C.cols // nelts), C.cols):
                C[m, n] += A[m, k] * B[k, n]


# Simplify the code by using the builtin vectorize function
# from Functional import vectorize
fn matmul_vectorized_1(C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[nelts, dot](C.cols)


# Parallelize the code by using the builtin parallelize function
# from Functional import parallelize
fn matmul_parallelized(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[nelts, dot](C.cols)

    parallelize[calc_row](rt, C.rows)


# Perform 2D tiling on the iteration space defined by end_x and end_y.
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    # Note: this assumes that ends are multiples of the tiles.
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)


# Use the above tile function to perform tiled matmul.
fn matmul_tiled_parallelized(C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):

                @parameter
                fn dot[
                    nelts: Int,
                ](n: Int):
                    C.store[nelts](
                        m,
                        n + x,
                        C.load[nelts](m, n + x)
                        + A[m, k] * B.load[nelts](k, n + x),
                    )

                vectorize[nelts, dot](tile_x)

        # We hardcode the tile factor to be 4.
        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, tile_size](A.cols, C.cols)

    parallelize[calc_row](rt, C.rows)


# Unroll the vectorized loop by a constant factor.
# from Functional import vectorize_unroll
fn matmul_tiled_unrolled_parallelized(
    C: Matrix, A: Matrix, B: Matrix
):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):

                @parameter
                fn dot[
                    nelts: Int,
                ](n: Int):
                    C.store[nelts](
                        m,
                        n + x,
                        C.load[nelts](m, n + x)
                        + A[m, k] * B.load[nelts](k, n + x),
                    )

                # Vectorize by nelts and unroll by tile_x/nelts
                # Here unroll factor is 4
                vectorize_unroll[nelts, tile_x // nelts, dot](tile_x)

        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, tile_size](A.cols, C.cols)

    parallelize[calc_row](rt, C.rows)

def matmul_untyped(C, A, B):
    for m in range(C.rows):
        for k in range(A.cols):
            for n in range(C.cols):
                C[m, n] += A[m, k] * B[k, n]

fn matrix_getitem(self: object, i: object) raises -> object:
    return self.value[i]


fn matrix_setitem(self: object, i: object, value: object) raises -> object:
    self.value[i] = value
    return None


fn matrix_append(self: object, value: object) raises -> object:
    self.value.append(value)
    return None


fn matrix_init(rows: Int, cols: Int) raises -> object:
    let value = object([])
    return object(
        Attr("value", value), Attr("__getitem__", matrix_getitem), Attr("__setitem__", matrix_setitem), 
        Attr("rows", rows), Attr("cols", cols), Attr("append", matrix_append),
    )

def benchmark_matmul_untyped(M: Int, N: Int, K: Int):
    C = matrix_init(M, N)
    A = matrix_init(M, K)
    B = matrix_init(K, N)
    for i in range(M):
        c_row = object([])
        b_row = object([])
        a_row = object([])
        for j in range(N):
            c_row.append(0.0)
            b_row.append(random_float64(-5, 5))
            a_row.append(random_float64(-5, 5))
        C.append(c_row)
        B.append(b_row)
        A.append(a_row)

    @parameter
    fn test_fn():
        try:
            _ = matmul_tiled_parallelized(C, A, B)
        except:
            pass

    let secs = Float64(Benchmark().run[test_fn]()) / 1_000_000_000
    _ = (A, B, C)
    let gflops = ((2*M*N*K)/secs) / 1e9
    print(gflops, "GFLOP/s")

fn main():
    try:
        benchmark_matmul_untyped(128, 128, 128)
    except:
        return