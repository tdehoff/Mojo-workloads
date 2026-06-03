from std.sys import has_accelerator, argv
from std.math import ceildiv, exp, sqrt, erf
from std.time import monotonic
from std.atomic import Atomic

from std.gpu import block_dim, block_idx, thread_idx
from std.gpu.host import DeviceContext
from layout import Layout, LayoutTensor

comptime num_iter = 10
comptime natoms = 256
comptime ngauss = 3
comptime filepath = "../tests/he" + String(natoms)
comptime geom_layout = Layout.col_major(natoms, ngauss)

comptime pi: Float64 = 3.1415926535897931
comptime sqrtpi2: Float64 = pow(pi, -0.5) * 2.0
comptime dtol: Float64 = 1.0e-12
comptime rcut: Float64 = 1.0e-12
comptime tobohrs: Float64 = 1.889725987722
comptime layout = Layout.row_major(natoms, natoms)
comptime dtype = DType.float64

def read_file(xpnt: UnsafePointer[mut=True, Float64, ...],
             coef: UnsafePointer[mut=True, Float64, ...],
             geom: UnsafePointer[mut=True, Float64, ...]) raises -> Bool:
    var file = open(filepath, "r")
    var buff = file.read().splitlines()

    var line = buff[0].split()
    if line[0].__int__() != ngauss or line[1].__int__() != natoms:
        print("ERROR: Invalid ngauss or natoms")
        return False

    var curr_line = 1
    for i in range(ngauss):
        line = buff[curr_line].split()
        while len(line) == 0:
            curr_line += 1
            line = buff[curr_line].split()
        if len(line) == 2:
            xpnt[i] = line[0].__float__()
            coef[i] = line[1].__float__()
        curr_line += 1

    for i in range(natoms):
        line = buff[curr_line].split()
        while len(line) == 0:
            curr_line += 1
            line = buff[curr_line].split()
        for j in range(3):
            geom[j * natoms + i] = line[j].__float__()
        curr_line += 1

    return True

# SSSS integral function
def ssss(i: Int, j: Int, k: Int, l: Int, ngauss: Int,
        xpnt: UnsafePointer[mut=True, Float64, ...],
        coef: UnsafePointer[mut=True, Float64, ...],
        geom: LayoutTensor[dtype, geom_layout, MutAnyOrigin]) -> Float64:
    var eri: Float64 = 0.0
    for ib in range(ngauss):
        for jb in range(ngauss):
            var aij: Float64 = 1.0 / (xpnt[ib] + xpnt[jb])
            var dij: Float64 = coef[ib] * coef[jb] *
                  exp(-xpnt[ib] * xpnt[jb] * aij *
                      ((pow(geom[i, 0][0] - geom[j, 0][0], 2)) +
                       (pow(geom[i, 1][0] - geom[j, 1][0], 2)) +
                       (pow(geom[i, 2][0] - geom[j, 2][0], 2)))) * pow(aij, 1.5)

            if abs(dij) > dtol:
                var xij: Float64 = aij * (xpnt[ib] * geom[i, 0][0] + xpnt[jb] * geom[j, 0][0])
                var yij: Float64 = aij * (xpnt[ib] * geom[i, 1][0] + xpnt[jb] * geom[j, 1][0])
                var zij: Float64 = aij * (xpnt[ib] * geom[i, 2][0] + xpnt[jb] * geom[j, 2][0])

                for kb in range(ngauss):
                    for lb in range(ngauss):
                        var akl: Float64 = 1.0 / (xpnt[kb] + xpnt[lb])
                        var dkl: Float64 = dij * coef[kb] * coef[lb] *
                              exp(-xpnt[kb] * xpnt[lb] * akl *
                              ((pow(geom[k, 0][0] - geom[l, 0][0], 2)) +
                               (pow(geom[k, 1][0] - geom[l, 1][0], 2)) +
                               (pow(geom[k, 2][0] - geom[l, 2][0], 2)))) * pow(akl, 1.5)

                        if abs(dkl) > dtol:
                            var aijkl: Float64 = (xpnt[ib] + xpnt[jb]) *
                                    (xpnt[kb] + xpnt[lb]) /
                                    (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb])
                            var tt: Float64 = aijkl * (
                                pow(xij - akl * (xpnt[kb] * geom[k, 0][0] + xpnt[lb] * geom[l, 0][0]), 2)
                                + pow(yij - akl * (xpnt[kb] * geom[k, 1][0] + xpnt[lb] * geom[l, 1][0]), 2)
                                + pow(zij - akl * (xpnt[kb] * geom[k, 2][0] + xpnt[lb] * geom[l, 2][0]), 2))
                            var f0t: Float64 = sqrtpi2
                            if tt > rcut:
                                f0t = pow(tt, -0.5) * erf(sqrt(tt))
                            eri += dkl * f0t * sqrt(aijkl)
    return eri

def hartree_fock_kernel(ngauss: Int,
                       schwarz: UnsafePointer[Float64, MutAnyOrigin],
                       xpnt: UnsafePointer[Float64, MutAnyOrigin],
                       coef: UnsafePointer[Float64, MutAnyOrigin],
                       geom: LayoutTensor[dtype, geom_layout, MutAnyOrigin],
                       dens: LayoutTensor[dtype, layout, MutAnyOrigin],
                       fock: LayoutTensor[dtype, layout, MutAnyOrigin]):
    var ijkl = block_idx.x * block_dim.x + thread_idx.x
    var ij = sqrt(2 * ijkl)
    var n: Float64 = Float64(ij * ij + ij) / 2.0

    while n < Float64(ijkl):
        ij += 1
        n = Float64(ij * ij + ij) / 2.0
    var kl = ijkl - (ij * ij - ij) // 2
    if schwarz[ij] * schwarz[kl] > dtol:
        var i = sqrt(2 * ij) - 1
        n = Float64(i * i + i) / 2.0
        while n < Float64(ij):
            i += 1
            n = Float64(i * i + i) / 2.0
        var j = ij - (i * i - i) // 2

        var k = sqrt(2 * kl)
        n = Float64(k * k + k) / 2.0
        while n < Float64(kl):
            k += 1
            n = Float64(k * k + k) / 2.0
        var l = kl - (k * k - k) // 2
        i -= 1
        j -= 1
        k -= 1
        l -= 1

        var eri: Float64 = 0.0
        for ib in range(ngauss):
            for jb in range(ngauss):
                var aij: Float64 = 1.0 / (xpnt[ib] + xpnt[jb])
                var dij: Float64 = coef[ib] * coef[jb] *
                                   exp(-xpnt[ib] * xpnt[jb] * aij *
                                       (pow(geom[i, 0][0] - geom[j, 0][0], 2) +
                                        pow(geom[i, 1][0] - geom[j, 1][0], 2) +
                                        pow(geom[i, 2][0] - geom[j, 2][0], 2))) * pow(aij, 1.5)

                if abs(dij) > dtol:
                    var xij: Float64 = aij * (xpnt[ib] * geom[i, 0][0] + xpnt[jb] * geom[j, 0][0])
                    var yij: Float64 = aij * (xpnt[ib] * geom[i, 1][0] + xpnt[jb] * geom[j, 1][0])
                    var zij: Float64 = aij * (xpnt[ib] * geom[i, 2][0] + xpnt[jb] * geom[j, 2][0])

                    for kb in range(ngauss):
                        for lb in range(ngauss):
                            var akl: Float64 = 1.0 / (xpnt[kb] + xpnt[lb])
                            var dkl: Float64 = dij * coef[kb] * coef[lb] *
                                  exp(-xpnt[kb] * xpnt[lb] * akl *
                                  (pow(geom[k, 0][0] - geom[l, 0][0], 2) +
                                   pow(geom[k, 1][0] - geom[l, 1][0], 2) +
                                   pow(geom[k, 2][0] - geom[l, 2][0], 2))) * pow(akl, 1.5)
                            if abs(dkl) > dtol:
                                var aijkl: Float64 = (xpnt[ib] + xpnt[jb]) *
                                        (xpnt[kb] + xpnt[lb]) /
                                        (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb])
                                var tt: Float64 = aijkl * (
                                    pow(xij - akl * (xpnt[kb] * geom[k, 0][0] + xpnt[lb] * geom[l, 0][0]), 2)
                                    + pow(yij - akl * (xpnt[kb] * geom[k, 1][0] + xpnt[lb] * geom[l, 1][0]), 2)
                                    + pow(zij - akl * (xpnt[kb] * geom[k, 2][0] + xpnt[lb] * geom[l, 2][0]), 2))
                                var f0t: Float64 = sqrtpi2
                                if tt > rcut:
                                    f0t = pow(tt, -0.5) * erf(pow(tt, 0.5))
                                eri += dkl * f0t * pow(aijkl, 0.5)

        if i == j:
            eri *= 0.5
        if k == l:
            eri *= 0.5
        if i == k and j == l:
            eri *= 0.5

        _ = Atomic[dtype].fetch_add(fock.ptr + (i * natoms + j),
                             rebind[Scalar[dtype]](dens[k, l][0] * eri * 4.0))
        _ = Atomic[dtype].fetch_add(fock.ptr + (k * natoms + l),
                             rebind[Scalar[dtype]](dens[i, j][0] * eri * 4.0))
        _ = Atomic[dtype].fetch_add(fock.ptr + (i * natoms + k),
                             rebind[Scalar[dtype]](dens[j, l][0] * eri * -1))
        _ = Atomic[dtype].fetch_add(fock.ptr + (i * natoms + l),
                             rebind[Scalar[dtype]](dens[j, k][0] * eri * -1))
        _ = Atomic[dtype].fetch_add(fock.ptr + (j * natoms + k),
                             rebind[Scalar[dtype]](dens[i, l][0] * eri * -1))
        _ = Atomic[dtype].fetch_add(fock.ptr + (j * natoms + l),
                             rebind[Scalar[dtype]](dens[i, k][0] * eri * -1))

def main() raises:
    args = argv()
    csv_output = False

    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--csv":
            csv_output = True
        i += 1

    if not has_accelerator():
        print("No compatible GPU found")
    else:
        ctx = DeviceContext()

        if not csv_output:
            print("GPU:", ctx.name())
            print("Driver:", ctx.get_api_version())
            print("natoms:", natoms, "; ngauss:", ngauss)

        xpnt = ctx.enqueue_create_host_buffer[dtype](ngauss)
        coef = ctx.enqueue_create_host_buffer[dtype](ngauss)
        geom = ctx.enqueue_create_host_buffer[dtype](natoms * ngauss)

        if read_file(xpnt.unsafe_ptr(), coef.unsafe_ptr(), geom.unsafe_ptr()):
            for i in range(natoms):
                geom[0 * natoms + i] *= tobohrs
                geom[1 * natoms + i] *= tobohrs
                geom[2 * natoms + i] *= tobohrs

            geom_tensor = LayoutTensor[dtype, geom_layout, MutAnyOrigin](geom.unsafe_ptr())

            # Doesn't work on AMD
            # dens = ctx.enqueue_create_host_buffer[dtype](natoms * natoms).enqueue_fill(0.1)
            dens = ctx.enqueue_create_host_buffer[dtype](natoms * natoms)
            ctx.synchronize()

            for i in range(natoms * natoms):
                dens[i] = 0.1

            for i in range(natoms):
                dens[i * natoms + i] = 1.0

            nn = ((natoms * natoms) + natoms) // 2
            schwarz = ctx.enqueue_create_host_buffer[dtype](nn + 1)

            for i in range(ngauss):
                coef[i] = coef[i] * pow((2.0 * xpnt[i]), 0.75)

            var ij = 0
            var eri: Float64 = 0.0
            for i in range(natoms):
                for j in range(i + 1):
                    ij = ij + 1
                    eri = ssss(i, j, i, j, ngauss, xpnt.unsafe_ptr(), coef.unsafe_ptr(), geom_tensor)
                    schwarz[ij] = sqrt(abs(eri))

            d_geom = ctx.enqueue_create_buffer[dtype](natoms * ngauss)
            ctx.enqueue_copy(dst_buf=d_geom, src_buf=geom)
            d_geom_tensor = LayoutTensor[dtype, geom_layout, MutAnyOrigin](d_geom.unsafe_ptr())

            d_dens = ctx.enqueue_create_buffer[dtype](natoms * natoms)
            ctx.enqueue_copy(dst_buf=d_dens, src_buf=dens)
            d_dens_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](d_dens.unsafe_ptr())

            d_fock = ctx.enqueue_create_buffer[dtype](natoms * natoms)
            d_fock.enqueue_fill(0.0)
            d_fock_tensor = LayoutTensor[dtype, layout, MutAnyOrigin](d_fock.unsafe_ptr())

            d_schwarz = ctx.enqueue_create_buffer[dtype](nn + 1)
            ctx.enqueue_copy(dst_buf=d_schwarz, src_buf=schwarz)

            d_coef = ctx.enqueue_create_buffer[dtype](ngauss)
            with d_coef.map_to_host() as h_coef:
                for i in range(ngauss):
                    h_coef[i] = coef[i]

            d_xpnt = ctx.enqueue_create_buffer[dtype](ngauss)
            with d_xpnt.map_to_host() as h_xpnt:
                for i in range(ngauss):
                    h_xpnt[i] = xpnt[i]
            ctx.synchronize()

            nnnn = ((nn * nn) + nn) // 2
            block_size = 256
            n_blocks = ceildiv(nnnn, block_size)

            # Warmup call
            ctx.enqueue_function[hartree_fock_kernel](
                ngauss, d_schwarz.unsafe_ptr(),
                d_xpnt.unsafe_ptr(), d_coef.unsafe_ptr(),
                d_geom_tensor, d_dens_tensor,
                d_fock_tensor,
                grid_dim = (n_blocks, 1, 1),
                block_dim = (block_size, 1, 1)
            )
            ctx.synchronize()

            var erep: Float64 = 0.0
            with d_fock.map_to_host() as final_fock, d_dens.map_to_host() as final_dens:
                for i in range(natoms):
                    for j in range(natoms):
                        erep += final_fock[i * natoms + j] * final_dens[i * natoms + j]

            if not csv_output:
                print("2e- energy =", erep * 0.5)
                # Expected result for he8 test
                # print("expected: 11.585212581525834")
            else:
                print("backend,GPU,natoms,ngauss,exec_time_ms")

                for _ in range(num_iter):
                    start = monotonic()
                    ctx.enqueue_function[hartree_fock_kernel](
                        ngauss, d_schwarz.unsafe_ptr(),
                        d_xpnt.unsafe_ptr(), d_coef.unsafe_ptr(),
                        d_geom_tensor, d_dens_tensor,
                        d_fock_tensor,
                        grid_dim = (n_blocks, 1, 1),
                        block_dim = (block_size, 1, 1)
                    )
                    ctx.synchronize()
                    end = monotonic()

                    elapsed = end - start
                    print("Mojo,", ctx.name(), ",", natoms, ",", ngauss, ",", Float64(elapsed) / 1e6)
