/******************************************************************************
Copyright (c) 2022 Advanced Micro Devices, Inc. (AMD)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
******************************************************************************/

/*
 Problem statement:

    Compute the Laplacian of a grid function u on a equidistantly spaced grid using a finite difference approximation

    f = \delta u

    Input parameters:

    ./laplacian <nx> <ny> <nz> <blk_x> <blk_y> <blk_z>

Based on AMD lab notes : Finite difference method – Laplacian part 1

https://github.com/amd/amd-lab-notes/blob/release/finite-difference/examples/kernel1.hpp
*/

#include <iostream>
#include <math.h>
#include <stdio.h>
#include <assert.h>
#include <vector>
#include <string>
#include "kernel.hpp"

using precision = float;
using namespace std;

char precision_str[] = "float";


const uint32_t TBSize = 256;
const uint32_t L = 1024;
const uint32_t NUM_ITER = 1000;
bool csv_output = false;

// HIP error check
#define HIP_CHECK(stat)                                           \
{                                                                 \
    if(stat != hipSuccess)                                        \
    {                                                             \
        std::cerr << "HIP error: " << hipGetErrorString(stat) <<  \
        " in file " << __FILE__ << ":" << __LINE__ << std::endl;  \
        exit(-1);                                                 \
    }                                                             \
}

template <typename T>
__global__ void test_function_kernel(T *u, int nx, int ny, int nz,
                                     T hx, T hy, T hz) {

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
    HIP_CHECK( hipGetLastError() );
}

int main(int argc, char **argv)
{
    // Default thread block sizes
    int BLK_X = TBSize;
    int BLK_Y = 1;
    int BLK_Z = 1;

    // Default problem size
    size_t nx = L, ny = L, nz = L;

    // precision tolerance = 3e-6;

    if (argc > 1) nx = atoi(argv[1]);
    if (argc > 2) ny = atoi(argv[2]);
    if (argc > 3) nz = atoi(argv[3]);
    if (argc > 4) BLK_X = atoi(argv[4]);
    if (argc > 5) BLK_Y = atoi(argv[5]);
    if (argc > 6) BLK_Z = atoi(argv[6]);
    if (argc > 7 && string(argv[7]) == "--csv") csv_output = true;

    // Theoretical fetch and write sizes:
    size_t theoretical_fetch_size = (nx * ny * nz - 8 - 4 * (nx - 2) - 4 * (ny - 2) - 4 * (nz - 2) ) * sizeof(precision);
    size_t theoretical_write_size = ((nx - 2) * (ny - 2) * (nz - 2)) * sizeof(precision);
    size_t datasize = theoretical_fetch_size + theoretical_write_size;

    int device;
    HIP_CHECK( hipGetDevice(&device) );
    hipDeviceProp_t props;
    HIP_CHECK( hipGetDeviceProperties(&props, device) );

    if (csv_output) {
        cout << "backend,GPU,precision,L,blk_x,blk_y,blk_z,BW_GBs" << endl;
    }
    else {
        cout << props.name << endl;
        cout << "Precision: double" << endl;
        cout << "nx,ny,nz = " << nx << ", " << ny << ", " << nz << endl;
        cout << "block sizes = " << BLK_X << ", " << BLK_Y << ", " << BLK_Z << endl;
        cout << "Theoretical fetch size (GB): " << theoretical_fetch_size * 1e-9 << endl;
        cout << "Theoretical write size (GB): " << theoretical_write_size * 1e-9 << endl;
    }

    size_t numbytes = nx * ny * nz * sizeof(precision);

    precision *d_u, *d_f;
    HIP_CHECK( hipMalloc((void**)&d_u, numbytes) );
    HIP_CHECK( hipMalloc((void**)&d_f, numbytes) );

    // Grid spacings
    precision hx = 1.0 / (nx - 1);
    precision hy = 1.0 / (ny - 1);
    precision hz = 1.0 / (nz - 1);

    // Initialize test function: 0.5 * (x * (x - 1) + y * (y - 1) + z * (z - 1))
    test_function(d_u, nx, ny, nz, hx, hy, hz);

    // Compute Laplacian (1/2) (x(x-1) + y(y-1) + z(z-1)) = 3 for all interior points
    laplacian(d_f, d_u, nx, ny, nz, BLK_X, BLK_Y, BLK_Z, hx, hy, hz);

    // Timing
    float total_elapsed = 0;
    float elapsed;
    hipEvent_t start, stop;
    HIP_CHECK( hipEventCreate(&start) );
    HIP_CHECK( hipEventCreate(&stop)  );

    for (int iter = 0; iter < NUM_ITER; ++iter) {
        // Flush cache
        HIP_CHECK( hipDeviceSynchronize()                     );
        HIP_CHECK( hipEventRecord(start)                      );
        laplacian(d_f, d_u, nx, ny, nz, BLK_X, BLK_Y, BLK_Z, hx, hy, hz);
        HIP_CHECK( hipGetLastError()                          );
        HIP_CHECK( hipEventRecord(stop)                       );
        HIP_CHECK( hipEventSynchronize(stop)                  );
        HIP_CHECK( hipEventElapsedTime(&elapsed, start, stop) );
        if (csv_output) {
            printf("HIP,%s,%s,%zu,%d,%d,%d,%g\n", props.name, precision_str,
            nx, BLK_X, BLK_Y, BLK_Z, datasize / elapsed / 1e6);
        }
        total_elapsed += elapsed;
    }

    // Effective memory bandwidth
    if (!csv_output) {
        printf("Laplacian kernel took: %g ms, effective memory bandwidth: %g GB/s \n\n",
                total_elapsed / NUM_ITER,
                datasize * NUM_ITER / total_elapsed / 1e6
                );
    }

    // Clean up
    HIP_CHECK( hipFree(d_f) );
    HIP_CHECK( hipFree(d_u) );

    return 0;
}