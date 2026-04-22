from std.sys import argv, has_accelerator
from std.sys.info import size_of
from std.collections import List
from std.math import ceildiv
from std.time import monotonic
from std.memory import stack_allocation

from std.gpu import block_dim, block_idx, thread_idx, grid_dim, barrier
from std.gpu.host import DeviceContext
from std.gpu.memory import AddressSpace, load
from std.python import Python

comptime dtype = DType.float64
comptime SIZE = 1 << 25
comptime num_iter = 10   # 1000 + 1 warmup runs
comptime TBSize = 1024

comptime initA: Scalar[dtype] = 0.1
comptime initB: Scalar[dtype] = 0.2
comptime initC: Scalar[dtype] = 0.0
comptime startScalar: Scalar[dtype] = 0.4
comptime DOT_READ_DWORDS_PER_LANE = 4

def init_kernel(
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    c: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    initA: Scalar[dtype],
    initB: Scalar[dtype],
    initC: Scalar[dtype],
):
    var i = block_dim.x * block_idx.x + thread_idx.x
    a[i] = initA
    b[i] = initB
    c[i] = initC

def copy_kernel(
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    c: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    var i = block_dim.x * block_idx.x + thread_idx.x
    c[i] = a[i]

def mul_kernel(
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    c: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    scalar: Scalar[dtype],
):
    var i = block_dim.x * block_idx.x + thread_idx.x
    b[i] = scalar * c[i]

def add_kernel(
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    c: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    var i = block_dim.x * block_idx.x + thread_idx.x
    c[i] = a[i] + b[i]

def triad_kernel(
    a: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    c: UnsafePointer[Scalar[dtype], MutAnyOrigin],
    scalar: Scalar[dtype],
):
    var i = block_dim.x * block_idx.x + thread_idx.x
    a[i] = b[i] + scalar * c[i]

def dot_kernel[size: Int](
    a: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    b: UnsafePointer[Scalar[dtype], ImmutAnyOrigin],
    sums: UnsafePointer[Scalar[dtype], MutAnyOrigin],
):
    var tb_sum = stack_allocation[
        TBSize,
        Scalar[dtype],
        address_space = AddressSpace.SHARED,
    ]()
    var i = block_dim.x * block_idx.x + thread_idx.x
    var local_tid = thread_idx.x
    tb_sum[local_tid] = 0

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

def main() raises:
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
    comptime if not SIZE % TBSize == 0:
        print("Array size must be a multiple of", TBSize)

    comptime if not has_accelerator():
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
        if not DOT_READ_DWORDS_PER_LANE * size_of[UInt]() < size_of[Scalar[dtype]]():
            dot_elements_per_lane = DOT_READ_DWORDS_PER_LANE * size_of[UInt]() // size_of[Scalar[dtype]]()

        # Round dot_num_blocks up to next multiple of (TBSIZE * dot_elements_per_lane)
        var dot_num_blocks = ceildiv(SIZE, (TBSize * dot_elements_per_lane))

        # Array of partial sums for dor kernel
        sums = ctx.enqueue_create_host_buffer[dtype](dot_num_blocks)
        # sums_ptr = sums.unsafe_ptr()
        d_sums = ctx.enqueue_create_buffer[dtype](dot_num_blocks)
        d_sums_ptr = d_sums.unsafe_ptr()

        # Initialize a, b, c
        ctx.enqueue_function[init_kernel, init_kernel](
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
            ctx.enqueue_function[copy_kernel, copy_kernel](
                a_ptr, b_ptr,
                grid_dim = (ceildiv(SIZE, TBSize)),
                block_dim = TBSize
            )
            ctx.synchronize()
            end = monotonic()
            kernel_timings[0][i] = Float32(end - start)

            # Test mul:
            start = monotonic()
            ctx.enqueue_function[mul_kernel, mul_kernel](
                b_ptr, c_ptr, startScalar,
                grid_dim = (ceildiv(SIZE, TBSize)),
                block_dim = TBSize
            )
            ctx.synchronize()
            end = monotonic()
            kernel_timings[1][i] = Float32(end - start)

            # Test add:
            start = monotonic()
            ctx.enqueue_function[add_kernel, add_kernel](
                a_ptr, b_ptr, c_ptr,
                grid_dim = (ceildiv(SIZE, TBSize)),
                block_dim = TBSize
            )
            ctx.synchronize()
            end = monotonic()
            kernel_timings[2][i] = Float32(end - start)

            # Test triad:
            start = monotonic()
            ctx.enqueue_function[triad_kernel, triad_kernel](
                a_ptr, b_ptr, c_ptr, startScalar,
                grid_dim = (ceildiv(SIZE, TBSize)),
                block_dim = TBSize
            )
            ctx.synchronize()
            end = monotonic()
            kernel_timings[3][i] = Float32(end - start)

            # Test dot:
            start = monotonic()
            ctx.enqueue_function[dot_kernel[SIZE], dot_kernel[SIZE]](
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
        var bytes_per_elem = size_of[Scalar[dtype]]()
        # Copy: 2N, Mul: 2N, Add: 3N, Triad: 3N, Dot: 2N
        var kernel_data: List[Int] = [
            2 * SIZE * bytes_per_elem,
            2 * SIZE * bytes_per_elem,
            3 * SIZE * bytes_per_elem,
            3 * SIZE * bytes_per_elem,
            2 * SIZE * bytes_per_elem,
        ]

        if csv_output:
            print("backend,GPU,precision,vec_size,routine,BW_GBs")
            for i in range(5):
                for k in range(num_iter):
                    print("Mojo,", ctx.name(), ",", dtype, ",", SIZE, ",", kernel_names[i], ",", kernel_data[i] / kernel_timings[i][k])
        else:
            print("Array size:", Float64(SIZE * bytes_per_elem) * 1e-6, "MB")
            print("Total size:", Float64(3 * SIZE * bytes_per_elem) * 1e-6, "MB")
            for i in range (5):
                print(kernel_names[i], ":")
                # Ignore the first result
                print("   Min (sec):", kernel_timings[i][1:].min() * 1e-9)
                print("   Max (sec):", kernel_timings[i][1:].max() * 1e-9)
                print("   Avg (sec):", kernel_timings[i][1:].mean() * 1e-9)
                print("   Bandwidth (GB/s):", kernel_data[i] / kernel_timings[i][1:].min())
