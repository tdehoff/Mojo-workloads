#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <iomanip>
#include <fstream>
#include <string>
#include <math.h>

#include <cuda.h>

const double pi = 3.1415926535897931;
const double sqrtpi2 = pow(pi,-0.5) * 2.0;
const double dtol = 1.0e-12;
const double rcut = 1.0e-12;
const double tobohrs = 1.889725987722;

bool csv_output = false;
int num_iter = 10;

__device__ const struct {
	double sqrtpi2;
	double dtol;
	double rcut;
} dev_consts = {1.12837916709551255856, dtol, rcut};

#define CUDA_CHECK(err)  __checkCudaErrors (err, __FILE__, __LINE__)

inline void __checkCudaErrors(cudaError err, const char *file, const int line )
{
  if(cudaSuccess != err){
	fprintf(stderr, "%s(%i) : CUDA Runtime API error %d: %s.\n",file, line, (int)err, cudaGetErrorString( err ) );
	 exit(-1);
  }
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

	CUDA_CHECK(cudaMallocManaged(xpnt, sizeof(double) * *ngauss));
	CUDA_CHECK(cudaMallocManaged(coef, sizeof(double) * *ngauss));
	CUDA_CHECK(cudaMallocManaged(geom, sizeof(double) * (3 * *natoms)));

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

	CUDA_CHECK(cudaMemcpy(*xpnt, x, *ngauss * sizeof(double), cudaMemcpyDefault));
	CUDA_CHECK(cudaMemcpy(*coef, c, *ngauss * sizeof(double), cudaMemcpyDefault));
	CUDA_CHECK(cudaMemcpy(*geom, g, (3 * *natoms) * sizeof(double), cudaMemcpyDefault));


	delete[] x;
	delete[] c;
	delete[] g;

}

__global__ void hartree_fock(uint64_t nnnn, uint32_t ngauss, uint32_t natoms,
		double *geom, double *xpnt, double *coef, double *dens,
		double *schwarz, double *fock)
{
	uint64_t ijkl = blockIdx.x * blockDim.x + threadIdx.x;
	uint64_t ij = (uint64_t)sqrtf(2*ijkl);
	double n = (ij*ij + ij) / 2;
	while (n < ijkl) {
		ij += 1;
		n = (ij*ij + ij) / 2;
	}
	uint64_t kl = ijkl - (ij * ij - ij) / 2;
	if (schwarz[ij] * schwarz[kl] > dtol) {
		uint64_t i = (uint64_t)sqrtf(2*ij)-1;
		n = (i * i + i) / 2;
		while (n < ij) {
			i += 1;
			n = (i * i + i) / 2;
		}
		uint64_t j = ij - (i * i - i) / 2;

		uint64_t k = (uint64_t)sqrtf(2*kl);
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
				//printf("aij=%.16lf\n", aij);
				double dij = coef[ib] * coef[jb] *
					exp(-xpnt[ib] * xpnt[jb] * aij *
							(pow(geom[i] - geom[j], 2)
							 + pow(geom[1*natoms + i] - geom[1*natoms + j], 2)
							 + pow(geom[2*natoms + i] - geom[2*natoms + j], 2)))
					* pow(aij, 1.5);
				if (abs(dij) > dtol) {
					//printf("index=%.16lf\n", xpnt[ib]);
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
								/*
								   if (threadIdx.x == 1) {
								   printf("eri=%.16lf\n", eri);
								   }
								   */
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
		atomicAdd(&fock[i*natoms + k], -dens[j*natoms + l]*eri);
		atomicAdd(&fock[i*natoms + l], -dens[j*natoms + k]*eri);
		atomicAdd(&fock[j*natoms + k], -dens[i*natoms + l]*eri);
		atomicAdd(&fock[j*natoms + l], -dens[i*natoms + k]*eri);
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
	CUDA_CHECK(cudaMallocManaged(&dens, sizeof(double) * (natoms * natoms)));
	CUDA_CHECK(cudaMallocManaged(&fock, sizeof(double) * (natoms * natoms)));
	CUDA_CHECK(cudaMallocManaged(&schwarz, sizeof(double) * nn + 1));

    int device;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp props;
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

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
			ij = ij + 1;
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
	cudaDeviceSynchronize();

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
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));

        for (int i = 0; i < num_iter; ++i) {
            CUDA_CHECK(cudaDeviceSynchronize());
            CUDA_CHECK(cudaEventRecord(start));
            hartree_fock<<<grid, threads>>>(nnnn, ngauss, natoms,
                    geom, xpnt, coef, dens,
                    schwarz,fock);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaEventRecord(stop));
            CUDA_CHECK(cudaEventSynchronize(stop));
            CUDA_CHECK(cudaEventElapsedTime(&elapsed, start, stop));

            if (csv_output) {
                printf("CUDA,%s,%d,%d,%f\n", props.name, natoms,
                ngauss, elapsed);
            }
        }
    }

	CUDA_CHECK(cudaFree(xpnt));
	CUDA_CHECK(cudaFree(coef));
	CUDA_CHECK(cudaFree(geom));
	CUDA_CHECK(cudaFree(dens));
	CUDA_CHECK(cudaFree(fock));
	CUDA_CHECK(cudaFree(schwarz));
}
