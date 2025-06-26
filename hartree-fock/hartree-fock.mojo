from gpu.host import DeviceContext
from sys import has_accelerator
from gpu.id import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from sys.info import sizeof
from math import ceildiv, exp, sqrt, erf, isqrt
from time import monotonic
from os.atomic import Atomic

alias natoms = 8
alias ngauss = 3
alias geom_layout = layout.col_major(natoms, ngauss)

alias pi: Float64 = 3.1415926535897931
alias sqrtpi2: Float64 = pow(pi, -0.5) * 2.0
alias dtol: Float64 = 1.0e-12
alias rcut: Float64 = 1.0e-12
alias tobohrs: Float64 = 1.889725987722
alias layout = Layout.col_major(natoms, natoms)
alias dtype = DType.float64

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
                      ((pow(geom[0, i] - geom[0, j], 2)) +
                       (pow(geom[1, i] - geom[1, j], 2)) +
                       (pow(geom[2, i] - geom[2, j], 2)))) * pow(aij, 1.5)

            if abs(dij) > dtol:
                xij = aij * (xpnt[ib] * geom[0, i] + xpnt[jb] * geom[1, j])
                yij = aij * (xpnt[ib] * geom[1, i] + xpnt[jb] * geom[2, j])
                zij = aij * (xpnt[ib] * geom[2, i] + xpnt[jb] * geom[2, j])

                for kb in range(ngauss):
                    for lb in range(ngauss):
                        akl = 1.0 / (xpnt[kb] + xpnt[lb])
                        dkl = dij * coef[kb] * coef[lb] *
                              exp(-xpnt[kb] * xpnt[lb] * akl *
                              ((pow(geom[0, k] - geom[0, l], 2)) +
                               (pow(geom[1, k] - geom[1, l], 2)) +
                               (pow(geom[2, k] - geom[2, l], 2)))) * pow(akl, 1.5)

                        if abs(dkl) > dtol:
                            aijkl = (xpnt[ib] + xpnt[jb]) *
                                    (xpnt[kb] + xpnt[lb]) /
                                    (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb])
                            tt = aijkl * (pow(xij - akl * (xpnt[kb] * geom[0, k] + xpnt[lb] * geom[0, l]), 2)
                                          + pow(yij - akl * (xpnt[kb] * geom[1, k] + xpnt[lb] * geom[1, l]), 2)
                                          + pow(zij - akl * (xpnt[kb] * geom[2, k] + xpnt[lb] * geom[2, l]), 2))
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
    var n: Float64 = (Float64)(ij * ij + ij) / 2
    while n < ijkl:
        ij += 1
        n = (Float64)(ij * ij + ij) / 2
    var kl = ijkl - (ij * ij - ij) // 2
    if schwarz[ij] * schwarz[kl] > dtol:
        var i = sqrt(2 * ij) - 1
        n = (Float64)(i * i + i) / 2
        while n < ij:
            i += 1
            n = (Float64)(i * i + i) / 2
        var j = ij - (i * i - i) // 2

        var k = sqrt(2 * kl)
        n = (k * k + k) / 2
        while n < kl:
            k += 1
            n = (k * k + k) / 2
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
                                       (pow(geom[0, i] - geom[0, j], 2) +
                                        pow(geom[1, i] - geom[1, j], 2) +
                                        pow(geom[2, i] - geom[2, j], 2)))) * pow(aij, 1.5)
                # print("aij =", aij, "dij =", dij)
                if abs(dij) > dtol:
                    xij = aij * (xpnt[ib] * geom[0, i] + xpnt[jb] * geom[0, j])
                    yij = aij * (xpnt[ib] * geom[1, i] + xpnt[jb] * geom[1, j])
                    zij = aij * (xpnt[ib] * geom[2, i] + xpnt[jb] * geom[2, j])
                    for kb in range(ngauss):
                        for lb in range(ngauss):
                            akl = 1.0 / (xpnt[kb] + xpnt[lb])
                            dkl = dij * coef[kb] * coef[lb] *
                                  exp(-xpnt[kb] * xpnt[lb] * akl *
                                  (pow(geom[0, k] - geom[0, l], 2) +
                                   pow(geom[1, k] - geom[1, l], 2) +
                                   pow(geom[2, k] - geom[2, l], 2))) * pow(akl, 1.5)
                            if abs(dkl) > dtol:
                                aijkl = (xpnt[ib] + xpnt[jb]) *
                                        (xpnt[kb] + xpnt[lb]) /
                                        (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb])
                                tt = aijkl * (pow(xij - akl * (xpnt[kb] * geom[0, k] + xpnt[lb] * geom[0, l]), 2)
                                               + pow(yij - akl * (xpnt[kb] * geom[1, k] + xpnt[lb] * geom[1, l]), 2)
                                               + pow(zij - akl * (xpnt[kb] * geom[2, k] + xpnt[lb] * geom[2, l]), 2))
                                # print("-> thread", ijkl, "tt =", tt, "aijkl =", aijkl)
                                f0t = sqrtpi2
                                if tt > rcut:
                                    f0t = Float64(pow(tt, -0.5) * erf(pow(tt, 0.5)))
                                    # print("THREAD", ijkl,"pow(tt, -0.5) =", pow(tt, -0.5), "erf(pow(tt, 0.5))", erf(pow(tt, 0.5)), "f0t =", f0t)
                                eri += Float64(dkl * f0t * pow(aijkl, 0.5))

                                if ijkl == 1:
                                    print("eri =", eri)
        if i == j:
            eri *= 0.5
        if k == l:
            eri *= 0.5
        if i == k and j == l:
            eri *= 0.5

        print("thread", ijkl, "eri =", eri)

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
    if not has_accelerator():
        print("No compatible GPU found")
    else:
        ctx = DeviceContext()
        print("GPU:", ctx.name())

        xpnt = [6.3624214, 1.1589230, 0.3136498]
        coef = [0.154328967295, 0.535328142282, 0.444634542185]

        geom = ctx.enqueue_create_host_buffer[dtype](natoms * ngauss)
        g_vals = [0.0000, 0.0000, 0.0000, 0.0000, 1.4000, 1.4000, 1.4000, 1.4000,
                  0.0000, 0.0000, 1.4000, 1.4000, 0.0000, 0.0000, 1.4000, 1.4000,
                  0.0000, 1.4000, 0.0000, 1.4000, 0.0000, 1.4000, 0.0000, 1.4000]

        for i in range(natoms):
            g_vals[0 * natoms + i] *= tobohrs
            g_vals[1 * natoms + i] *= tobohrs
            g_vals[2 * natoms + i] *= tobohrs

        for i in range(natoms * ngauss):
            geom[i] = g_vals[i]

        geom_tensor = LayoutTensor[dtype, geom_layout](geom)

        dens = ctx.enqueue_create_host_buffer[dtype](natoms * natoms).enqueue_fill(0.1)
        ctx.synchronize()
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
            print(final_fock)
            print("             ")
            print(final_dens)
            for i in range(natoms):
                for j in range(natoms):
                    erep += final_fock[i * natoms + j] * final_dens[i * natoms + j]
        print("2e- energy =", erep * 0.5)
        print("expected: 11.585212581525834")