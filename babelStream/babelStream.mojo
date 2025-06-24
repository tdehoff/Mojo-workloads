from gpu.host import DeviceContext
from sys import has_accelerator
from gpu import block_dim, block_idx, thread_idx, grid_dim, barrier
from sys.info import sizeof
from math import ceildiv
from time import monotonic
from sys import argv
from memory import stack_allocation
from os.atomic import Atomic
from gpu.memory import AddressSpace, load
from collections import List
from python import Python

alias dtype = DType.float64
# Default size of 2^25
alias SIZE = 33554432
alias num_runs = 100
alias TBSize = 1024
alias initA: Scalar[dtype] = 0.1
alias initB: Scalar[dtype] = 0.2
alias initC: Scalar[dtype] = 0.0
alias startScalar: Scalar[dtype] = 0.4

fn init_kernel(
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    c: UnsafePointer[Scalar[dtype]],
    initA: Scalar[dtype],
    initB: Scalar[dtype],
    initC: Scalar[dtype],
):
    var i = block_dim.x * block_idx.x + thread_idx.x
    a[i] = initA
    b[i] = initB
    c[i] = initC

fn copy_kernel(
    a: UnsafePointer[Scalar[dtype]],
    c: UnsafePointer[Scalar[dtype]],
):
    var i = block_dim.x * block_idx.x + thread_idx.x
    c[i] = a[i]

fn mul_kernel(
    b: UnsafePointer[Scalar[dtype]],
    c: UnsafePointer[Scalar[dtype]],
    scalar: Scalar[dtype],
):
    var i = block_dim.x * block_idx.x + thread_idx.x
    b[i] = scalar * c[i]

fn add_kernel(
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    c: UnsafePointer[Scalar[dtype]],
):
    var i = block_dim.x * block_idx.x + thread_idx.x
    c[i] = a[i] + b[i]

fn triad_kernel(
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    c: UnsafePointer[Scalar[dtype]],
    scalar: Scalar[dtype],
):
    var i = block_dim.x * block_idx.x + thread_idx.x
    a[i] = b[i] + scalar * c[i]

fn dot_kernel[size: Int](
    a: UnsafePointer[Scalar[dtype]],
    b: UnsafePointer[Scalar[dtype]],
    output: UnsafePointer[Scalar[dtype]],
):
    var tb_sum = stack_allocation[
        TBSize,
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
    ]()
    var global_tid = block_dim.x * block_idx.x + thread_idx.x
    var local_tid = thread_idx.x
    threads_in_grid = TBSize * grid_dim.x
    sum: Scalar[dtype] = 0

    for i in range(global_tid, size, threads_in_grid):
        sum += a[i] * b[i]
    tb_sum[local_tid] = sum
    barrier()

    var offset = block_dim.x // 2
    while offset > 0:
        if local_tid < offset:
            tb_sum[local_tid] += tb_sum[local_tid + offset]
        barrier()
        offset //= 2

    if local_tid == 0:
        _ =  Atomic.fetch_add(output, tb_sum[0])

def main():
    np = Python.import_module("numpy")
    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        ctx = DeviceContext()
        print("GPU:", ctx.name())
        d_a = ctx.enqueue_create_buffer[dtype](SIZE)
        d_b = ctx.enqueue_create_buffer[dtype](SIZE)
        d_c = ctx.enqueue_create_buffer[dtype](SIZE)
        d_sum = ctx.enqueue_create_buffer[dtype](1).enqueue_fill(0)
        a_ptr = d_a.unsafe_ptr()
        b_ptr = d_b.unsafe_ptr()
        c_ptr = d_c.unsafe_ptr()
        out_ptr = d_sum.unsafe_ptr()

        # Initialize a, b, c
        ctx.enqueue_function[init_kernel](
            a_ptr, b_ptr, c_ptr,
            initA, initB, initC,
            grid_dim = (ceildiv(SIZE, TBSize)),
            block_dim = TBSize
        )
        ctx.synchronize()

        # Timing:
        # total_elapsed: UInt = 0
        kernel_timings = np.zeros(Python.tuple(5, num_runs), dtype="float32")

        for i in range(num_runs):
            # Test copy:
                start = monotonic()
                ctx.enqueue_function[copy_kernel](
                    a_ptr, b_ptr,
                    grid_dim = (ceildiv(SIZE, TBSize)),
                    block_dim = TBSize
                )
                ctx.synchronize()
                end = monotonic()
                kernel_timings[0][i] = Float32(end - start)

                # Test mul:
                start = monotonic()
                ctx.enqueue_function[mul_kernel](
                    b_ptr, c_ptr, startScalar,
                    grid_dim = (ceildiv(SIZE, TBSize)),
                    block_dim = TBSize
                )
                ctx.synchronize()
                end = monotonic()
                kernel_timings[1][i] = Float32(end - start)

                # Test add:
                start = monotonic()
                ctx.enqueue_function[add_kernel](
                    a_ptr, b_ptr, c_ptr,
                    grid_dim = (ceildiv(SIZE, TBSize)),
                    block_dim = TBSize
                )
                ctx.synchronize()
                end = monotonic()
                kernel_timings[2][i] = Float32(end - start)

                # Test triad:
                start = monotonic()
                ctx.enqueue_function[triad_kernel](
                    a_ptr, b_ptr, c_ptr, startScalar,
                    grid_dim = (ceildiv(SIZE, TBSize)),
                    block_dim = TBSize
                )
                ctx.synchronize()
                end = monotonic()
                kernel_timings[3][i] = Float32(end - start)

                # Test dot:
                start = monotonic()
                ctx.enqueue_function[dot_kernel[SIZE]](
                    a_ptr, b_ptr, out_ptr,
                    grid_dim = (ceildiv(SIZE, TBSize)),
                    block_dim = TBSize
                )
                ctx.synchronize()
                end = monotonic()
                kernel_timings[4][i] = Float32(end - start)


        kernel_names = ["Copy", "Mul", "Add", "Triad", "Dot"]
        # Copy: 2N, Mul: 2N, Add: 3N, Triad: 3N, Dot: 2N
        kernel_data = List[Int64](
            2 * SIZE * sizeof[Scalar[dtype]](),
            2 * SIZE * sizeof[Scalar[dtype]](),
            3 * SIZE * sizeof[Scalar[dtype]](),
            3 * SIZE * sizeof[Scalar[dtype]](),
            2 * SIZE * sizeof[Scalar[dtype]](),
        )

        print("Array size:", SIZE * sizeof[Scalar[dtype]]() * 1e-6, "MB")
        print("Total size:", 3 * SIZE * sizeof[Scalar[dtype]]() * 1e-6, "MB")

        for i in range (5):
            print(kernel_names[i], ":")
            print("   Min (sec):", kernel_timings[i][1:].min() * 1e-9)
            print("   Max (sec):", kernel_timings[i][1:].max() * 1e-9)
            print("   Avg (sec):", kernel_timings[i][1:].mean() * 1e-9)
            print("   Bandwidth (GB/s):", kernel_data[i] / kernel_timings[i][1:].min())
