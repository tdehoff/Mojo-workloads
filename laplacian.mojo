from gpu.host import DeviceContext
from sys import has_accelerator
from gpu.id import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from sys.info import sizeof
from math import ceildiv
from time import monotonic

alias precision = Float32
alias float_dtype = DType.float32
alias L = 256
alias steps = 100
alias layout = Layout.row_major(L, L, L)
alias BLK_X = 16
alias BLK_Y = 16
alias BLK_Z = 4

fn laplacian_kernel(
    f: LayoutTensor[float_dtype, layout, MutableAnyOrigin],
    u: LayoutTensor[float_dtype, layout, MutableAnyOrigin],
    nx: Int,
    ny: Int,
    nz: Int,
    invhx2: precision,
    invhy2: precision,
    invhz2: precision,
    invhxyz2: precision,
):
    var i = thread_idx.x + block_idx.x * block_dim.x
    var j = thread_idx.y + block_idx.y * block_dim.y
    var k = thread_idx.z + block_idx.z * block_dim.z

    if i >= 1 and i < nx-1 and
       j >= 1 and j < ny-1 and
       k >= 1 and k < nz-1:
        f[i, j, k] = u[i, j, k] * invhxyz2
               + (u[i - 1, j    , k    ] + u[i + 1, j    , k    ]) * invhx2
               + (u[i    , j - 1, k    ] + u[i    , j + 1, k    ]) * invhy2
               + (u[i    , j    , k - 1] + u[i    , j    , k + 1]) * invhz2

fn test_function_kernel(
    u: LayoutTensor[float_dtype, layout, MutableAnyOrigin],
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
        x: precision = i * hx
        y: precision = j * hy
        z: precision = k * hz
        Lx: precision = nx * hx
        Ly: precision = ny * hy
        Lz: precision = nz * hz
        u[i, j, k] = c * x * (x - Lx) + c * y * (y - Ly) + c * z * (z - Lz)

def main():
    @parameter
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        nx = L
        ny = L
        nz = L
        print("------------------------------")
        print("L =", L, "; Block dimensions:", BLK_X, BLK_Y, BLK_Z)

        ctx = DeviceContext()
        print("GPU:", ctx.name())
        d_u = ctx.enqueue_create_buffer[float_dtype](nx * ny * nz)
        d_f = ctx.enqueue_create_buffer[float_dtype](nx * ny * nz)
        u_tensor = LayoutTensor[float_dtype, layout](d_u)
        f_tensor = LayoutTensor[float_dtype, layout](d_f)

        # Grid point spacings
        hx = 1.0 / (nx - 1)
        hy = 1.0 / (ny - 1)
        hz = 1.0 / (nz - 1)
        # Initialize test function: 0.5 * (x * (x - 1) + y * (y - 1) + z * (z - 1))
        ctx.enqueue_function[test_function_kernel](
            u_tensor, nx, ny, nz, hx, hy, hz,
            grid_dim = (ceildiv(nx, BLK_X), ny, nz),
            block_dim = (BLK_X, 1)
        )
        ctx.synchronize()

        # Compute laplacian
        invhx2 = 1.0 / hx / hx
        invhy2 = 1.0 / hy / hy
        invhz2 = 1.0 / hz / hz
        invhxyz2 = -2.0 * (invhx2 + invhy2 + invhz2)
        # Warmup call
        ctx.enqueue_function[laplacian_kernel](
            f_tensor, u_tensor, nx, ny, nz,
            invhx2, invhy2, invhz2, invhxyz2,
            grid_dim = (ceildiv(nx, BLK_X), ceildiv(ny, BLK_Y), ceildiv(nz, BLK_Z)),
            block_dim = (BLK_X, BLK_Y, BLK_Z)
        )
        ctx.synchronize()

        # Timing:
        total_elapsed: UInt = 0

        for _ in range(steps):
            start = monotonic()
            ctx.enqueue_function[laplacian_kernel](
                f_tensor, u_tensor, nx, ny, nz,
                invhx2, invhy2, invhz2, invhxyz2,
                grid_dim = (ceildiv(nx, BLK_X), ceildiv(ny, BLK_Y), ceildiv(nz, BLK_Z)),
                block_dim = (BLK_X, BLK_Y, BLK_Z)
            )
            ctx.synchronize()
            end = monotonic()
            total_elapsed += (end - start)

        # Effective memory bandwidth
        theoretical_fetch_size = (nx * ny * nz - 8 - 4 * (nx - 2) - 4 * (ny - 2) - 4 * (nz - 2)) * sizeof[precision]()
        theoretical_write_size = ((nx - 2) * (ny - 2) * (nz - 2)) * sizeof[precision]()
        print("Theoretical fetch size (GB):", theoretical_fetch_size * 1e-9)
        print("Theoretical fetch size (GB):", theoretical_write_size * 1e-9)
        datasize = theoretical_fetch_size + theoretical_write_size
        print("Average kernel time:", total_elapsed / 1e9 / steps, "s")
        print("Effective memory bandwidth:", datasize * steps / total_elapsed, "GB/s")

        # # Copy result to host
        # h_f = ctx.enqueue_create_host_buffer[float_dtype](nx * ny * nz)
        # ctx.enqueue_copy(dst_buf=h_f, src_buf=d_f)
        # ctx.synchronize()
        # print("h_f:", h_f)
