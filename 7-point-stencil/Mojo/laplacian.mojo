# Based on AMD lab notes : Finite difference method – Laplacian part 1
# https://github.com/amd/amd-lab-notes/blob/release/finite-difference/examples/kernel1.hpp

from std.sys import argv, has_accelerator
from std.sys.info import size_of
from std.math import ceildiv
from std.time import monotonic
from layout import LayoutTensor, Layout

from std.gpu.host import DeviceContext
from std.gpu import block_dim, block_idx, thread_idx

comptime L = 512
comptime num_iter = 1000
comptime TBSize = 512

comptime precision = Float32
comptime dtype = DType.float32
comptime layout = Layout.row_major(L, L, L)

def laplacian_kernel(
    f: LayoutTensor[dtype, layout, MutAnyOrigin],
    u: LayoutTensor[dtype, layout, MutAnyOrigin],
    nx: Int,
    ny: Int,
    nz: Int,
    invhx2: precision,
    invhy2: precision,
    invhz2: precision,
    invhxyz2: precision,
):
    var k = Int(thread_idx.x + block_idx.x * block_dim.x)
    var j = Int(thread_idx.y + block_idx.y * block_dim.y)
    var i = Int(thread_idx.z + block_idx.z * block_dim.z)

    if i > 0 and i < nx - 1 and
       j > 0 and j < ny - 1 and
       k > 0 and k < nz - 1:
        f[i, j, k] = u[i, j, k] * invhxyz2
               + (u[i - 1, j    , k    ] + u[i + 1, j    , k    ]) * invhx2
               + (u[i    , j - 1, k    ] + u[i    , j + 1, k    ]) * invhy2
               + (u[i    , j    , k - 1] + u[i    , j    , k + 1]) * invhz2

def test_function_kernel(
    u: LayoutTensor[dtype, layout, MutAnyOrigin],
    nx: Int,
    ny: Int,
    nz: Int,
    hx: precision,
    hy: precision,
    hz: precision,
):
    var i = thread_idx.x + block_idx.x * block_dim.x
    var j = thread_idx.y + block_idx.y * block_dim.y
    var k = thread_idx.z + block_idx.z * block_dim.z

    # Exit if this thread is outside the boundary
    if i < nx and j < ny and k < nz:
        c: precision = 0.5
        x: precision = precision(i) * hx
        y: precision = precision(j) * hy
        z: precision = precision(k) * hz
        Lx: precision = precision(nx) * hx
        Ly: precision = precision(ny) * hy
        Lz: precision = precision(nz) * hz
        u[j, i, k] = c * x * (x - Lx) + c * y * (y - Ly) + c * z * (z - Lz)

def main() raises:
    args = argv()
    BLK_X: Int = TBSize
    BLK_Y: Int = 1
    BLK_Z: Int = 1
    csv_output = False

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--block" and i + 3 < len(args):
            try:
                BLK_X = args[i + 1].__int__()
                BLK_Y = args[i + 2].__int__()
                BLK_Z = args[i + 3].__int__()
            except e:
                raise e^
            i += 3
        elif arg == "--csv":
            csv_output = True
        i += 1

    comptime if not has_accelerator():
        print("No compatible GPU found")
    else:
        nx = L
        ny = L
        nz = L

        ctx = DeviceContext()
        if not csv_output:
            print("------------------------------")
            print("L =", L, "; Block dimensions:", BLK_X, BLK_Y, BLK_Z)
            print("GPU:", ctx.name())
            print("Driver:", ctx.get_api_version())

        # Data for bandwidth calculation:
        theoretical_fetch_size = (nx * ny * nz - 8 - 4 * (nx - 2) - 4 * (ny - 2) - 4 * (nz - 2)) * size_of[precision]()
        theoretical_write_size = ((nx - 2) * (ny - 2) * (nz - 2)) * size_of[precision]()
        datasize = theoretical_fetch_size + theoretical_write_size

        d_u = ctx.enqueue_create_buffer[dtype](nx * ny * nz)
        d_f = ctx.enqueue_create_buffer[dtype](nx * ny * nz)
        u_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](d_u)
        f_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](d_f)

        # Grid point spacings
        hx = 1.0 / precision(nx - 1)
        hy = 1.0 / precision(ny - 1)
        hz = 1.0 / precision(nz - 1)

        # Initialize test function: 0.5 * (x * (x - 1) + y * (y - 1) + z * (z - 1))
        ctx.enqueue_function[test_function_kernel, test_function_kernel](
            u_tensor, nx, ny, nz, hx, hy, hz,
            grid_dim = (ceildiv(nx, BLK_X), ny, nz),
            block_dim = (BLK_X, 1)
        )
        ctx.synchronize()

        # Compute laplacian
        var invhx2: precision = 1.0 / hx / hx
        var invhy2: precision = 1.0 / hy / hy
        var invhz2: precision = 1.0 / hz / hz
        var invhxyz2: precision = -2.0 * (invhx2 + invhy2 + invhz2)

        # Warmup call
        ctx.enqueue_function[laplacian_kernel, laplacian_kernel](
            f_tensor, u_tensor, nx, ny, nz,
            invhx2, invhy2, invhz2, invhxyz2,
            grid_dim = (ceildiv(nx, BLK_X), ceildiv(ny, BLK_Y), ceildiv(nz, BLK_Z)),
            block_dim = (BLK_X, BLK_Y, BLK_Z)
        )
        ctx.synchronize()

        # Timing:
        total_elapsed: UInt = 0
        if csv_output:
            print("backend,GPU,precision,L,blk_x,blk_y,blk_z,BW_GBs")

        for _ in range(num_iter):
            start = monotonic()
            ctx.enqueue_function[laplacian_kernel, laplacian_kernel](
                f_tensor, u_tensor, nx, ny, nz,
                invhx2, invhy2, invhz2, invhxyz2,
                grid_dim = (ceildiv(nx, BLK_X), ceildiv(ny, BLK_Y), ceildiv(nz, BLK_Z)),
                block_dim = (BLK_X, BLK_Y, BLK_Z)
            )
            ctx.synchronize()
            end = monotonic()

            elapsed = end - start
            var bw_gbs = Float64(datasize) / Float64(elapsed)
            if csv_output:
                print("mojo,", ctx.name(), ",", dtype, ",", L, ",", BLK_X, ",", BLK_Y, ",", BLK_Z, ",", bw_gbs)

            total_elapsed += elapsed

        if not csv_output:
            print("Theoretical fetch size (GB):", Float64(theoretical_fetch_size) * 1e-9)
            print("Theoretical fetch size (GB):", Float64(theoretical_write_size) * 1e-9)
            print("Average kernel time:", Float64(total_elapsed) / 1e6 / num_iter, "ms")
            print("Effective memory bandwidth:", Float64(datasize * num_iter) / Float64(total_elapsed), "GB/s")
