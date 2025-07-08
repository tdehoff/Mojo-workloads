/*
Problem statement:

    Compute the Laplacian of a grid function u on a equidistantly spaced grid using a finite difference approximation

    f = \delta u

    Input parameters:

    ./laplacian <blk_x> <blk_y> <blk_z>

Based on Claire Winogrodzki's work:

https://github.com/JuliaORNL/JACC-repro/tree/main/7-point-stencil/NVIDIA
*/

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include "kernel.cu"

using precision = double;
using namespace std;

const int TBSize = 256;
const int L = 512;
const int NUM_ITER = 1000;
bool verbose = true;

// CUDA error check
#define CUDA_CHECK(stat)                                           \
{                                                                 \
    if(stat != cudaSuccess)                                        \
    {                                                             \
        std::cerr << "CUDA error: " << cudaGetErrorString(stat) <<  \
        " in file " << __FILE__ << ":" << __LINE__ << std::endl;  \
        exit(-1);                                                 \
    }                                                             \
}

template <typename T>
__global__ void test_function_kernel(T *u, int nx, int ny, int nz, T hx, T hy, T hz) {

    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int j = threadIdx.y + blockIdx.y * blockDim.y;
    int k = threadIdx.z + blockIdx.z * blockDim.z;

    // Exit if this thread is outside the boundary
    if (i >= nx ||
        j >= ny ||
        k >= nz)
        return;

    size_t pos = i + nx * (j +  ny * k);

    T c = 0.5;
    T x = i*hx;
    T y = j*hy;
    T z = k*hz;
    T Lx = nx*hx;
    T Ly = ny*hy;
    T Lz = nz*hz;
    u[pos] = c * x * (x - Lx) + c * y * (y - Ly) + c * z * (z - Lz);
}

template <typename T>
void test_function(T *d_f, int nx, int ny, int nz, T hx, T hy, T hz) {

    dim3 block(TBSize, 1);
    dim3 grid((nx - 1) / block.x + 1, ny, nz);

    test_function_kernel<<<grid, block>>>(d_f, nx, ny, nz, hx, hy, hz);
    CUDA_CHECK( cudaGetLastError() );
}

int main(int argc, char **argv){

    // Default thread block sizes
    int BLK_X = TBSize;
    int BLK_Y = 1;
    int BLK_Z = 1;

    // Default problem size
    size_t nx = L, ny = L, nz = L;

    precision tolerance = 3e-6;

    if (argc > 1) nx = atoi(argv[1]);
    if (argc > 2) ny = atoi(argv[2]);
    if (argc > 3) nz = atoi(argv[3]);
    if (argc > 4) BLK_X = atoi(argv[4]);
    if (argc > 5) BLK_Y = atoi(argv[5]);
    if (argc > 6) BLK_Z = atoi(argv[6]);

    cout << "Precision: double" << endl;

    cout << "nx,ny,nz = " << nx << ", " << ny << ", " << nz << endl;
    cout << "block sizes = " << BLK_X << ", " << BLK_Y << ", " << BLK_Z << endl;

    // Theoretical fetch and write sizes:
    size_t theoretical_fetch_size = (nx * ny * nz - 8 - 4 * (nx - 2) - 4 * (ny - 2) - 4 * (nz - 2) ) * sizeof(precision);
    size_t theoretical_write_size = ((nx - 2) * (ny - 2) * (nz - 2)) * sizeof(precision);

    std::cout << "Theoretical fetch size (GB): " << theoretical_fetch_size * 1e-9 << std::endl;
    std::cout << "Theoretical write size (GB): " << theoretical_write_size * 1e-9 << std::endl;

    size_t numbytes = nx * ny * nz * sizeof(precision);

    // Input and output arrays
    precision *d_u, *d_f;

    // Allocate on device
    CUDA_CHECK( cudaMalloc((void**)&d_u, numbytes) );
    CUDA_CHECK( cudaMalloc((void**)&d_f, numbytes) );

    // Grid point spacings
    precision hx = 1.0 / (nx - 1), hy = 1.0 / (ny - 1), hz = 1.0 / (nz - 1);

    // Initialize test function: 0.5 * (x * (x - 1) + y * (y - 1) + z * (z - 1))
    test_function(d_u, nx, ny, nz, hx, hy, hz);

    // Compute Laplacian (1/2) (x(x-1) + y(y-1) + z(z-1)) = 3 for all interior points
    laplacian(d_f, d_u, nx, ny, nz, BLK_X, BLK_Y, BLK_Z, hx, hy, hz);

    // Timing
    float total_elapsed = 0;
    float elapsed;
    cudaEvent_t start, stop;
    CUDA_CHECK( cudaEventCreate(&start) );
    CUDA_CHECK( cudaEventCreate(&stop)  );

    if (verbose) {
        cout << "Kernel execution times (ms):" << endl;
    }

    for (int iter = 0; iter < NUM_ITER; ++iter) {
        // Flush cache
        CUDA_CHECK( cudaDeviceSynchronize()                     );
        CUDA_CHECK( cudaEventRecord(start)                      );
        laplacian(d_f, d_u, nx, ny, nz, BLK_X, BLK_Y, BLK_Z, hx, hy, hz);
        CUDA_CHECK( cudaGetLastError()                          );
        CUDA_CHECK( cudaEventRecord(stop)                       );
        CUDA_CHECK( cudaEventSynchronize(stop)                  );
        CUDA_CHECK( cudaEventElapsedTime(&elapsed, start, stop) );
        if (verbose) {
            printf("%f\n", elapsed);
        }
        total_elapsed += elapsed;
    }

    // Effective memory bandwidth
    size_t datasize = theoretical_fetch_size + theoretical_write_size;
    printf("Laplacian kernel took: %g ms, effective memory bandwidth: %g GB/s \n",
            total_elapsed / NUM_ITER,
            datasize * NUM_ITER / total_elapsed / 1e6
            );

    // Clean up
    CUDA_CHECK( cudaFree(d_f) );
    CUDA_CHECK( cudaFree(d_u) );

    return 0;
}