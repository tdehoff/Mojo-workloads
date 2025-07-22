#include "hip/hip_runtime.h"

#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <string>
#include <math.h>

const double pi = 3.1415926535897931;
const double sqrtpi2 = pow(pi,-0.5) * 2.0;
const double dtol = 1.0e-10;
const double rcut = 1.0e-12;
const double tobohrs = 1.889725987722;

bool csv_output = false;
int num_iter = 10;

__device__ const struct {
	double sqrtpi2;
	double dtol;
	double rcut;
} dev_consts = {1.12837916709551255856, dtol, rcut};

#define HIP_CHECK(cmd) \
{\
    hipError_t error  = cmd;\
    if (error != hipSuccess) { \
        fprintf(stderr, "HIP error: '%s'(%d) at %s:%d\n",\
	        hipGetErrorString(error), error,__FILE__, __LINE__); \
        exit(EXIT_FAILURE);\
    }\
}


double ssss(uint32_t i, uint32_t j, uint32_t k, uint32_t l,
		       uint32_t ngauss, double *xpnt, double *coef,
		       double *geom, uint32_t natoms) {
	double eri = 0.0;
	double aij, dij, akl, dkl, aijkl;
	double xij, yij, zij;
	double f0t, tt;

	for (int ib = 0; ib < ngauss; ib++) {
	     for (int jb = 0; jb < ngauss; jb++) {
		  aij = 1.0 / (xpnt[ib] + xpnt[jb]);
		  dij = coef[ib] * coef[jb] *
			exp(-xpnt[ib] * xpnt[jb] * aij *
			    ((pow(geom[i] - geom[j], 2))
			    + (pow(geom[natoms + i] - geom[natoms + j], 2))
			    + (pow(geom[2*natoms + i] - geom[2*natoms + j], 2))))
			* pow(aij,1.5);
		  if (abs(dij) > dtol) {
		      xij = aij * (xpnt[ib] * geom[i] + xpnt[jb] * geom[j]);
		      yij = aij * (xpnt[ib] * geom[natoms + i] + xpnt[jb] * geom[natoms + j]);
		      zij = aij * (xpnt[ib] * geom[2*natoms + i] + xpnt[jb] * geom[2*natoms + j]);

		      for (int kb = 0; kb < ngauss; kb++) {
		           for(int lb = 0; lb < ngauss; lb++) {
				   akl = 1.0 / (xpnt[kb] + xpnt[lb]);
				   dkl = dij * coef[kb] * coef[lb] *
					 exp(-xpnt[kb] * xpnt[lb] * akl
				           * ((pow(geom[k] - geom[l], 2))
					       + (pow(geom[natoms + k] - geom[natoms + l],2))
					       + (pow(geom[2*natoms + k] - geom[2*natoms + l],2))))
					  * pow(akl, 1.5);

			           if (abs(dkl) > dtol) {
				       aijkl = (xpnt[ib] + xpnt[jb])
						* (xpnt[kb] + xpnt[lb])
						/ (xpnt[ib] + xpnt[jb]
					           + xpnt[kb] + xpnt[lb]);
				       tt = aijkl * (pow(xij - akl * (xpnt[kb] * geom[k] + xpnt[lb] * geom[l]), 2)
				                     + pow(yij - akl * (xpnt[kb] * geom[natoms + k] + xpnt[lb] * geom[natoms + l]), 2)
				                     + pow(zij - akl * (xpnt[kb] * geom[2*natoms + k] + xpnt[lb] * geom[2*natoms + l]), 2));
				       f0t = sqrtpi2;
				       if (tt > rcut) {
				       	   f0t = pow(tt, -0.5) * erf(sqrtf(tt));
				       }
				       eri += (dkl * f0t * sqrtf(aijkl));
				   }
			   }
		      }

		  }

	     }
	}
	return eri;

}

void read_file(char *filename, uint32_t* ngauss, uint32_t* natoms,
	       double** xpnt, double** coef, double** geom) {

	std::ifstream inp(filename);

	if (!inp) {
            std::cout << "ERROR: Could not read file!" << std::endl;
	    return ;
	}

	if (!(inp >> *ngauss)) {
	    std::cout << "ERROR: Could not read # of gaussians"
		      << std::endl;
	    return ;
	}
	if (!(inp >> *natoms)) {

	    std::cout << "ERROR: Could not read # of gaussians"
		      << std::endl;
	    return ;
	}

	HIP_CHECK(hipMallocManaged(xpnt, sizeof(double) * *ngauss));
	HIP_CHECK(hipMallocManaged(coef, sizeof(double) * *ngauss));
	HIP_CHECK(hipMallocManaged(geom, sizeof(double) * (3 * *natoms)));

	double *x = new double[*ngauss];
	double *c = new double[*ngauss];
	double *g = new double[3 * *natoms];

	for (int i = 0; i < *ngauss; i++) {
	     inp >> x[i];
	     inp >> c[i];
	}

	for (int i = 0; i < *natoms; i++) {
	     for (int j = 0; j < 3; j++) {
	          inp >> g[j* *natoms + i];
	     }
	}

	HIP_CHECK(hipMemcpy(*xpnt, x, *ngauss * sizeof(double), hipMemcpyDefault));
	HIP_CHECK(hipMemcpy(*coef, c, *ngauss * sizeof(double), hipMemcpyDefault));
	HIP_CHECK(hipMemcpy(*geom, g, (3 * *natoms) * sizeof(double), hipMemcpyDefault));


	delete[] x;
	delete[] c;
	delete[] g;

}

__global__ void hartree_fock(uint64_t nnnn, uint32_t ngauss, uint32_t natoms,
		double *geom, double *xpnt, double *coef, double *dens,
		double *schwarz, double *fock)
{
	uint64_t ijkl = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t ij = (uint64_t)sqrt(2*ijkl);
	double n = (ij*ij + ij) / 2;
	while (n < ijkl) {
		ij += 1;
		n = (ij*ij + ij) / 2;
	}
	uint64_t kl = ijkl - (ij * ij - ij) / 2;
	if (schwarz[ij] * schwarz[kl] > dtol) {
		uint64_t i = (uint64_t)sqrt(2*ij)-1;
		n = (i * i + i) / 2;
		while (n < ij) {
			i += 1;
			n = (i * i + i) / 2;
		}
		uint64_t j = ij - (i * i - i) / 2;

		uint64_t k = (uint64_t)sqrt(2*kl);
		n = (k * k + k) / 2;
		while (n < kl) {
			k += 1;
			n = (k * k + k ) / 2;
		}
		uint64_t l = kl - (k * k - k) / 2;
		i-= 1;
		j-= 1;
		k-= 1;
		l-= 1;

		double eri = 0.0;
		for (int ib = 0; ib < ngauss; ib++) {
			for (int jb = 0; jb < ngauss; jb++) {
				double aij = 1.0 / (xpnt[ib] + xpnt[jb]);
				double dij = coef[ib] * coef[jb] *
					exp(-xpnt[ib] * xpnt[jb] * aij *
							(pow(geom[i] - geom[j], 2)
							 + pow(geom[1*natoms + i] - geom[1*natoms + j], 2)
							 + pow(geom[2*natoms + i] - geom[2*natoms + j], 2)))
					* pow(aij, 1.5);
				if (abs(dij) > dtol) {
					double xij = aij * (xpnt[ib] * geom[i] + xpnt[jb] * geom[j]);
					double yij = aij * (xpnt[ib] * geom[1*natoms + i] + xpnt[jb] * geom[1*natoms + j]);
					double zij = aij * (xpnt[ib] * geom[2*natoms + i] + xpnt[jb] * geom[2*natoms + j]);
					for (int kb = 0; kb < ngauss; kb++) {
						for (int lb = 0; lb < ngauss; lb++) {
							double akl = 1.0 / (xpnt[kb] + xpnt[lb]);
							double dkl = dij * coef[kb] * coef[lb] *
								exp(-xpnt[kb] * xpnt[lb] * akl *
										(pow(geom[k] - geom[l], 2)
										 + pow(geom[natoms + k] - geom[natoms + l], 2)
										 + pow(geom[2*natoms + k] - geom[2*natoms + l], 2))) * pow(akl, 1.5);
							if (abs(dkl) > dtol) {
								double aijkl = (xpnt[ib] + xpnt[jb])
									* (xpnt[kb] + xpnt[lb])
									/ (xpnt[ib] + xpnt[jb] + xpnt[kb] + xpnt[lb]);
								double tt = aijkl * (pow(xij - akl * (xpnt[kb] * geom[k] + xpnt[lb] * geom[l]), 2)
										+ pow(yij - akl * (xpnt[kb] * geom[natoms + k] + xpnt[lb] * geom[natoms + l]), 2)
										+ pow(zij - akl * (xpnt[kb] * geom[2*natoms + k] + xpnt[lb] * geom[2*natoms + l]), 2));
								double f0t = dev_consts.sqrtpi2;
								if (tt > dev_consts.rcut) {
									f0t = (pow(tt, -0.5) * erf(sqrt(tt)));
								}
								eri += dkl * f0t * sqrtf(aijkl);
							}

						}

					}
				}

			}
		}
		if (i == j) {
			eri *= 0.5;
		}
		if (k == l) {
			eri *= 0.5;
		}
		if (i ==k && j == l) {
			eri *= 0.5;
		}
		atomicAdd(&fock[i*natoms + j], dens[k*natoms + l]*eri*4.0);
		atomicAdd(&fock[k*natoms + l], dens[i*natoms + j]*eri*4.0);
		atomicSub(&fock[i*natoms + k], dens[j*natoms + l]*eri);
		atomicSub(&fock[i*natoms + l], dens[j*natoms + k]*eri);
		atomicSub(&fock[j*natoms + k], dens[i*natoms + l]*eri);
		atomicSub(&fock[j*natoms + l], dens[i*natoms + k]*eri);
	}
}


int main(int argc, char **argv){
	if (argc < 2) {
		std::cout << "ERROR: Please provide an input file" << std::endl;
		return 1;
	}

    if (argc > 2 && std::string(argv[2]) == "--csv") csv_output = true;

    uint32_t natoms, ngauss;
	double *xpnt, *coef, *geom, *fock, *dens, *schwarz;

	read_file(argv[1], &ngauss, &natoms, &xpnt, &coef, &geom);

	uint32_t nn = (((natoms * natoms) + natoms) / 2);
	HIP_CHECK(hipMallocManaged(&dens, sizeof(double) * (natoms * natoms)));
	HIP_CHECK(hipMallocManaged(&fock, sizeof(double) * (natoms * natoms)));
	HIP_CHECK(hipMallocManaged(&schwarz, sizeof(double) * nn + 1));

    int device;
    HIP_CHECK(hipGetDevice(&device));
    hipDeviceProp_t props;
    HIP_CHECK(hipGetDeviceProperties(&props, device));

	for (int i = 0; i < natoms; i++){
		for (int j = 0; j < natoms; j++){
			dens[i*natoms + j] = 0.1;
		}
		dens[i*natoms + i] = 1.0;
	}


	for (int i = 0; i < ngauss; i++){
		coef[i] = coef[i] * pow((2.0 * xpnt[i]),0.75);
	}

	for (int i = 0; i < natoms; i++){
		geom[0 * natoms + i] *= tobohrs;
		geom[1 * natoms + i] *= tobohrs;
		geom[2 * natoms + i] *= tobohrs;
	}


	for (int i = 0; i < natoms; i++){
		for (int j = 0; j < natoms; j++){
			fock[i*natoms + j] = 0.0;
		}
	}

	uint32_t ij = 0;
	double eri = 0.0;
	for (uint32_t i = 0; i < natoms; i++) {
		for (uint32_t j = 0; j <= i; j++) {
			ij+=1;
			eri = ssss(i,j,i,j,ngauss,xpnt,coef,geom, natoms);
			schwarz[ij] = sqrt(abs(eri));
		}
	}

	uint64_t nnnn = ((nn * nn) + nn) / 2;
	uint32_t blk_size = 256;
	uint32_t n_blks = (nnnn + blk_size - 1) / blk_size;

	dim3 threads(blk_size, 1, 1);
	dim3 grid(n_blks, 1, 1);

    // Warmup call
	hartree_fock<<<grid, threads>>>(nnnn, ngauss, natoms,
			geom, xpnt, coef, dens,
			schwarz,fock);
	hipDeviceSynchronize();

	double erep = 0.0;
	for (uint32_t i = 0; i < natoms; i++){
		for (uint32_t j = 0; j < natoms; j++){
			erep += fock[i*natoms +j] * dens[i*natoms + j];
		}
	}

    if (!csv_output) {
    	printf("2e- energy=%.16lf\n", erep*0.5);
    }
    else {
        printf("backend,GPU,natoms,ngauss,exec_time_ms\n");

        // Timing
        float elapsed;
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));

        for (int i = 0; i < num_iter; ++i) {
            HIP_CHECK(hipDeviceSynchronize());
            HIP_CHECK(hipEventRecord(start));
            hartree_fock<<<grid, threads>>>(nnnn, ngauss, natoms,
                    geom, xpnt, coef, dens,
                    schwarz,fock);
            HIP_CHECK(hipGetLastError());
            HIP_CHECK(hipEventRecord(stop));
            HIP_CHECK(hipEventSynchronize(stop));
            HIP_CHECK(hipEventElapsedTime(&elapsed, start, stop));

            if (csv_output) {
                printf("HIP,%s,%d,%d,%f\n", props.name, natoms,
                ngauss, elapsed);
            }
        }
    }

	HIP_CHECK(hipFree(xpnt));
	HIP_CHECK(hipFree(coef));
	HIP_CHECK(hipFree(geom));
	HIP_CHECK(hipFree(dens));
	HIP_CHECK(hipFree(fock));
	HIP_CHECK(hipFree(schwarz));
}
