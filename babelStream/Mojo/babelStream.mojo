from sys import argv, has_accelerator
from sys.info import sizeof
from collections import List
from math import ceildiv
from time import monotonic
from memory import stack_allocation

from gpu import block_dim, block_idx, thread_idx, grid_dim, barrier
from gpu.host import DeviceContext
from gpu.memory import AddressSpace, load
from python import Python

alias dtype = DType.float64
# Default size of 2^25
alias SIZE = pow(2, 25)    # input size, must be a multiple of 1024
alias num_iter = 1001      # 1000 + 1 warmup runs
alias TBSize = 1024

alias initA: Scalar[dtype] = 0.1
alias initB: Scalar[dtype] = 0.2
alias initC: Scalar[dtype] = 0.0
alias startScalar: Scalar[dtype] = 0.4
alias DOT_READ_DWORDS_PER_LANE = 4

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
    sums: UnsafePointer[Scalar[dtype]],
):
    var tb_sum = stack_allocation[
        TBSize,
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
    ]()
    var i = block_dim.x * block_idx.x + thread_idx.x
    var local_tid = thread_idx.x

    threads_in_grid = block_dim.x * grid_dim.x
    while i < SIZE:
        tb_sum[local_tid] += a[i] * b[i]
        i += threads_in_grid

    var offset = block_dim.x // 2
    while offset > 0:
        barrier()
        if local_tid < offset:
            tb_sum[local_tid] += tb_sum[local_tid + offset]
        offset //= 2

    if local_tid == 0:
        sums[block_idx.x] = tb_sum[local_tid]

def main():
    csv_output = False
    np = Python.import_module("numpy")

    args = argv()
    verbose: Int = 0
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--csv":
            csv_output = True
        i += 1

    # The array size must be divisible by TBSIZE for kernel launches
    @parameter
    if not SIZE % TBSize == 0:
        print("Array size must be a multiple of", TBSize)

    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        ctx = DeviceContext()
        if not csv_output:
            print("GPU:", ctx.name())
            print("Driver:", ctx.get_api_version())

        d_a = ctx.enqueue_create_buffer[dtype](SIZE)
        d_b = ctx.enqueue_create_buffer[dtype](SIZE)
        d_c = ctx.enqueue_create_buffer[dtype](SIZE)
        a_ptr = d_a.unsafe_ptr()
        b_ptr = d_b.unsafe_ptr()
        c_ptr = d_c.unsafe_ptr()

        # Compute number of blocks for dot kernel
        var dot_elements_per_lane = 1
        if not DOT_READ_DWORDS_PER_LANE * sizeof[UInt]() < sizeof[Scalar[dtype]]():
            dot_elements_per_lane = DOT_READ_DWORDS_PER_LANE * sizeof[UInt]() // sizeof[Scalar[dtype]]()

        # Round dot_num_blocks up to next multiple of (TBSIZE * dot_elements_per_lane)
        var dot_num_blocks = ceildiv(SIZE, (TBSize * dot_elements_per_lane))

        # Array of partial sums for dor kernel
        sums = ctx.enqueue_create_host_buffer[dtype](dot_num_blocks)
        # sums_ptr = sums.unsafe_ptr()
        d_sums = ctx.enqueue_create_buffer[dtype](dot_num_blocks)
        d_sums_ptr = d_sums.unsafe_ptr()

        # Initialize a, b, c
        ctx.enqueue_function[init_kernel](
            a_ptr, b_ptr, c_ptr,
            initA, initB, initC,
            grid_dim = (ceildiv(SIZE, TBSize)),
            block_dim = TBSize
        )
        ctx.synchronize()

        # Timing:
        kernel_timings = np.zeros(Python.tuple(5, num_iter), dtype="float32")

        for i in range(num_iter):
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
                a_ptr, b_ptr, d_sums_ptr,
                grid_dim = dot_num_blocks,
                block_dim = TBSize
            )
            ctx.synchronize()
            sum: Scalar[dtype] = 0
            ctx.enqueue_copy(dst_buf=sums, src_buf=d_sums)
            ctx.synchronize()
            for i in range (dot_num_blocks):
                sum += sums[i]
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

        if csv_output:
            print("backend,GPU,precision,vec_size,routine,BW_GBs")
            for i in range(5):
                for k in range(num_iter):
                    print("Mojo,", ctx.name(), ",", dtype.__str__(), ",", SIZE, ",", kernel_names[i], ",", kernel_data[i] / kernel_timings[i][k])
        else:
            print("Array size:", SIZE * sizeof[Scalar[dtype]]() * 1e-6, "MB")
            print("Total size:", 3 * SIZE * sizeof[Scalar[dtype]]() * 1e-6, "MB")
            for i in range (5):
                print(kernel_names[i], ":")
                # Ignore the first result
                print("   Min (sec):", kernel_timings[i][1:].min() * 1e-9)
                print("   Max (sec):", kernel_timings[i][1:].max() * 1e-9)
                print("   Avg (sec):", kernel_timings[i][1:].mean() * 1e-9)
                print("   Bandwidth (GB/s):", kernel_data[i] / kernel_timings[i][1:].min())
