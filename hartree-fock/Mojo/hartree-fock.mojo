from sys import has_accelerator, argv
from sys.info import sizeof
from math import ceildiv, exp, sqrt, erf
from time import monotonic
from os.atomic import Atomic

from gpu.id import block_dim, block_idx, thread_idx
from gpu.host import DeviceContext
from layout import Layout, LayoutTensor

alias num_iter = 10
alias natoms = 256
alias ngauss = 3
alias filepath = "../tests/he" + String(natoms)
alias geom_layout = layout.col_major(natoms, ngauss)

alias pi: Float64 = 3.1415926535897931
alias sqrtpi2: Float64 = pow(pi, -0.5) * 2.0
alias dtol: Float64 = 1.0e-12
alias rcut: Float64 = 1.0e-12
alias tobohrs: Float64 = 1.889725987722
alias layout = Layout.row_major(natoms, natoms)
alias dtype = DType.float64

fn read_file(xpnt: UnsafePointer[Float64],
             coef: UnsafePointer[Float64],
             geom: UnsafePointer[Float64]) raises -> Bool:
    var file = open(filepath, "r")
    var buff = file.read().splitlines()

    # Read and verify ngauss and natoms
    var line = buff[0].split()
    if line[0].__int__() != ngauss or line[1].__int__() != natoms:
        print("ERROR: Invalid ngauss or natoms")
        return False

    var curr_line = 1
    # Read xpnt and coef
    for i in range(ngauss):
        line = buff[curr_line].split()
        # Skip empty lines
        while len(line) == 0:
            curr_line += 1
            line = buff[curr_line].split()
        if len(line) == 2:
            xpnt[i] = line[0].__float__()
            coef[i] = line[1].__float__()
        curr_line += 1

    # Read geom
    for i in range(natoms):
        line = buff[curr_line].split()
        # Skip empty lines
        while len(line) == 0:
            curr_line += 1
            line = buff[curr_line].split()
        for j in range(3):
            geom[j * natoms + i] = line[j].__float__()
        curr_line += 1

    return True

# SSSS integral function
fn ssss(i: Int, j: Int, k: Int, l: Int, ngauss: Int,
        xpnt: UnsafePointer[Float64], coef: UnsafePointer[Float64],
        geom: LayoutTensor[mut=True, dtype, geom_layout]) -> Float64:
    var eri: Float64 = 0.0
    for ib in range(ngauss):
        for jb in range(ngauss):
            aij = 1.0 / (xpnt[ib] + xpnt[jb])
            dij = coef[ib] * coef[jb] *
                  exp(-xpnt[ib] * xpnt[jb] * aij *
                      ((pow(geom[i, 0] - geom[j, 0], 2)) +
                       (pow(geom[i, 1] - geom[j, 1], 2)) +
                       (pow(geom[i, 2] - geom[j, 2], 2)))) * pow(aij, 1.5)

            if abs(dij) > dtol:
                xij = aij * (xpnt[ib] * geom[i, 0] + xpnt[jb] * geom[j, 0])
                yij = aij * (xpnt[ib] * geom[i, 1] + xpnt[jb] * geom[j, 1])
                zij = aij * (xpnt[ib] * geom[i, 2] + xpnt[jb] * geom[j, 2])

                for kb in range(ngauss):
                    for lb in range(ngauss):
                        akl = 1.0 / (xpnt[kb] + xpnt[lb])
                        dkl = dij * coef[kb] * coef[lb] *
                              exp(-xpnt[kb] * xpnt[lb] * akl *
                              ((pow(geom[k, 0] - geom[l, 0], 2)) +
                               (pow(geom[k, 1] - geom[l, 1], 2)) +
                               (pow(geom[k, 2] - geom[l, 2], 2)))) * pow(akl, 1.5)

                        if abs(dkl) > dtol:
                            aijkl = (xpnt[ib] + xpnt[jb]) *
                                    (xpnt[kb] + xpnt[lb]) /
                                    (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb])
                            tt = aijkl * (pow(xij - akl * (xpnt[kb] * geom[k, 0] + xpnt[lb] * geom[l, 0]), 2)
                                          + pow(yij - akl * (xpnt[kb] * geom[k, 1] + xpnt[lb] * geom[l, 1]), 2)
                                          + pow(zij - akl * (xpnt[kb] * geom[k, 2] + xpnt[lb] * geom[l, 2]), 2))
                            f0t = sqrtpi2
                            if tt > rcut:
                                f0t = (pow(tt, -0.5)[0]) * (erf(sqrt(tt))[0])
                            eri += Float64(dkl * f0t * sqrt(aijkl))
    return eri

fn hartree_fock_kernel(ngauss: Int, schwarz: UnsafePointer[Float64],
                       xpnt: UnsafePointer[Float64], coef: UnsafePointer[Float64],
                       geom: LayoutTensor[mut=True, dtype, geom_layout],
                       dens: LayoutTensor[mut=True, dtype, layout],
                       fock: LayoutTensor[mut=True, dtype, layout]):
    var ijkl = block_idx.x * block_dim.x + thread_idx.x
    var ij = sqrt(2 * ijkl)
    var n: Float64 = (ij * ij + ij) / 2.0

    while n < ijkl:
        ij += 1
        n = (ij * ij + ij) / 2.0
    var kl = ijkl - (ij * ij - ij) // 2
    if schwarz[ij] * schwarz[kl] > dtol:
        var i = sqrt(2 * ij) - 1
        n = (i * i + i) / 2.0
        while n < ij:
            i += 1
            n = (i * i + i) / 2.0
        var j = ij - (i * i - i) // 2

        var k = sqrt(2 * kl)
        n = (k * k + k) / 2.0
        while n < kl:
            k += 1
            n = (k * k + k) / 2.0
        var l = kl - (k * k - k) // 2
        i -= 1
        j -= 1
        k -= 1
        l -= 1

        var eri: Float64 = 0.0
        for ib in range(ngauss):
            for jb in range(ngauss):
                var aij: Float64 = 1.0 / (xpnt[ib] + xpnt[jb])
                var dij: Float64 = coef[ib] * coef[jb] * (Float64)(exp(-xpnt[ib] * xpnt[jb] * aij *
                                                          (pow(geom[i, 0] - geom[j, 0], 2) +
                                                           pow(geom[i, 1] - geom[j, 1], 2) +
                                                           pow(geom[i, 2] - geom[j, 2], 2)))) * pow(aij, 1.5)

                if abs(dij) > dtol:
                    xij = aij * (xpnt[ib] * geom[i, 0] + xpnt[jb] * geom[j, 0])
                    yij = aij * (xpnt[ib] * geom[i, 1] + xpnt[jb] * geom[j, 1])
                    zij = aij * (xpnt[ib] * geom[i, 2] + xpnt[jb] * geom[j, 2])

                    for kb in range(ngauss):
                        for lb in range(ngauss):
                            akl = 1.0 / (xpnt[kb] + xpnt[lb])
                            dkl = dij * coef[kb] * coef[lb] *
                                  exp(-xpnt[kb] * xpnt[lb] * akl *
                                  (pow(geom[k, 0] - geom[l, 0], 2) +
                                   pow(geom[k, 1] - geom[l, 1], 2) +
                                   pow(geom[k, 2] - geom[l, 2], 2))) * pow(akl, 1.5)
                            if abs(dkl) > dtol:
                                aijkl = (xpnt[ib] + xpnt[jb]) *
                                        (xpnt[kb] + xpnt[lb]) /
                                        (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb])
                                tt = aijkl * (pow(xij - akl * (xpnt[kb] * geom[k, 0] + xpnt[lb] * geom[l, 0]), 2)
                                               + pow(yij - akl * (xpnt[kb] * geom[k, 1] + xpnt[lb] * geom[l, 1]), 2)
                                               + pow(zij - akl * (xpnt[kb] * geom[k, 2] + xpnt[lb] * geom[l, 2]), 2))
                                f0t = sqrtpi2
                                if tt > rcut:
                                    f0t = Float64(pow(tt, -0.5) * erf(pow(tt, 0.5)))
                                eri += Float64(dkl * f0t * pow(aijkl, 0.5))

        if i == j:
            eri *= 0.5
        if k == l:
            eri *= 0.5
        if i == k and j == l:
            eri *= 0.5

        _ = Atomic.fetch_add(fock.ptr.offset(i * natoms + j),
                             rebind[Scalar[dtype]](dens[k, l] * eri * 4.0))
        _ = Atomic.fetch_add(fock.ptr.offset(k * natoms + l),
                             rebind[Scalar[dtype]](dens[i, j] * eri * 4.0))
        _ = Atomic.fetch_add(fock.ptr.offset(i * natoms + k),
                             rebind[Scalar[dtype]](dens[j, l] * eri * -1))
        _ = Atomic.fetch_add(fock.ptr.offset(i * natoms + l),
                             rebind[Scalar[dtype]](dens[j, k] * eri * -1))
        _ = Atomic.fetch_add(fock.ptr.offset(j * natoms + k),
                             rebind[Scalar[dtype]](dens[i, l] * eri * -1))
        _ = Atomic.fetch_add(fock.ptr.offset(j * natoms + l),
                             rebind[Scalar[dtype]](dens[i, k] * eri * -1))

def main():
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

            geom_tensor = LayoutTensor[dtype, geom_layout](geom)

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
            d_geom_tensor = LayoutTensor[dtype, geom_layout](d_geom)

            d_dens = ctx.enqueue_create_buffer[dtype](natoms * natoms)
            ctx.enqueue_copy(dst_buf=d_dens, src_buf=dens)
            d_dens_tensor = LayoutTensor[dtype, layout](d_dens)

            d_fock = ctx.enqueue_create_buffer[dtype](natoms * natoms).enqueue_fill(0.0)
            d_fock_tensor = LayoutTensor[dtype, layout](d_fock)

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
                    print("Mojo,", ctx.name(), ",", natoms, ",", ngauss, ",", elapsed / 1e6)