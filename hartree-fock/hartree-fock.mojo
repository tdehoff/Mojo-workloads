from gpu.host import DeviceContext
from sys import has_accelerator
from gpu.id import block_dim, block_idx, thread_idx
from layout import Layout, LayoutTensor
from sys.info import sizeof
from math import ceildiv, exp, sqrt, erf
from time import monotonic
from os.atomic import Atomic

alias natoms = 1024
alias ngauss = 3
alias txpnt = [6.3624214, 1.1589230, 0.3136498]
alias tcoef = [0.154328967295, 0.535328142282, 0.444634542185]

alias pi: Float64 = 3.1415926535897931
alias sqrtpi2: Float64 = pow(pi, -0.5) * 2.0
alias dtol: Float64 = 1.0e-12
alias rcut: Float64 = 1.0e-12
alias tobohrs: Float64 = 1.889725987722
alias layout = Layout.row_major(1024, 1024)         # change later
alias dtype = DType.float64

# SSSS integral function
fn ssss(i: Int, j: Int, k: Int, l: Int, ngauss: Int,
        xpnt: UnsafePointer[Float64], coef: UnsafePointer[Float64],
        geom: LayoutTensor[mut=True, dtype, layout]) -> Float64:
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
                                f0t = Float64(pow(tt, -0.5) * erf(sqrt(tt)))
                            eri += Float64(dkl * f0t * sqrt(aijkl))
    return eri

fn hartree_fock_kernel(ngauss: Int, schwarz: UnsafePointer[Float64],
                       xpnt: UnsafePointer[Float64], coef: UnsafePointer[Float64],
                       geom: LayoutTensor[mut=True, dtype, layout],
                       dens: LayoutTensor[mut=True, dtype, layout],
                       fock: LayoutTensor[mut=True, dtype, layout]):
    var ijkl = block_idx.x * block_dim.x + thread_idx.x
    var ij = sqrt(2 * ijkl)
    var n = (Float64)(ij * ij + ij) / 2
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

        eri: Float64 = 0.0
        for ib in range(ngauss):
            for jb in range(ngauss):
                aij = 1.0 / (xpnt[ib] + xpnt[jb])
                dij = coef[ib] * coef[jb] *
                      exp(-xpnt[ib] * xpnt[jb] * aij *
                          (pow(geom[0, i] - geom[0, j], 2) +
                           pow(geom[1, i] - geom[1, j], 2) +
                           pow(geom[2, i] - geom[2, j], 2))) * pow(aij, 1.5)
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
                                        (xpnt[ib] + xpnt[jb] + xpnt[kb] +
                                         xpnt[lb])
                                tt = aijkl * (pow(xij - akl * (xpnt[kb] * geom[0, k] + xpnt[lb] * geom[0, l]), 2) +
                                              pow(yij - akl * (xpnt[kb] * geom[1, k] + xpnt[lb] * geom[1, l]), 2) +
                                              pow(zij - akl * (xpnt[kb] * geom[2, k] + xpnt[lb] * geom[2, l]), 2))
                                f0t = sqrtpi2
                                if tt > rcut:
                                    f0t = Float64(pow(tt, -0.5) * erf(sqrt(tt)))
                                eri += Float64(dkl * f0t * sqrt(aijkl))
        if i == j:
            eri *= 0.5
        if k == l:
            eri *= 0.5
        if i== k and j == l:
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


