#include <cuda_runtime.h>

template <typename T>
__global__ void laplacian_kernel(T * f, const T * u, int nx, int ny, int nz, T invhx2, T invhy2, T invhz2, T invhxyz2) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    // Exit if this thread is on the boundary
    if (i == 0 || i >= nx - 1 ||
        j == 0 || j >= ny - 1 ||
        k == 0 || k >= nz - 1)
        return;

    // const int slice = nx * ny;
    // size_t pos = i + nx * j + slice * k;
    const int slice = ny * nz;
    size_t pos = k + nz * j + slice *i;

    // Compute the result of the stencil operation
    f[pos] = u[pos] * invhxyz2
           + (u[pos - 1]     + u[pos + 1]) * invhx2
           + (u[pos - nx]    + u[pos + nx]) * invhy2
           + (u[pos - slice] + u[pos + slice]) * invhz2;
}

template <typename T>
void laplacian(T *d_f, T *d_u, int nx, int ny, int nz, int BLK_X, int BLK_Y, int BLK_Z, T hx, T hy, T hz) {

    dim3 block(BLK_X, BLK_Y, BLK_Z);
    dim3 grid((nx - 1) / block.x + 1, (ny - 1) / block.y + 1, (nz - 1) / block.z + 1);
    T invhx2 = (T)1./hx/hx;
    T invhy2 = (T)1./hy/hy;
    T invhz2 = (T)1./hz/hz;
    T invhxyz2 = -2. * (invhx2 + invhy2 + invhz2);

    laplacian_kernel<<<grid, block>>>(d_f, d_u, nx, ny, nz, invhx2, invhy2, invhz2, invhxyz2);
}