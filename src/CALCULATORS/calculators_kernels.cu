
// CALCULATORS/calculators_kernels.cu
// All CUDA kernel definitions for calculator operations

#include "calculators.cuh"
#include "../INCLUDE/structs.cuh"
#include <cub/cub.cuh>
#include <cuda_runtime.h>
#include <iostream>
// Index macros for 2D grid with boundaries
#define IDX(i, j, Nb) ((i) * (Nb) + (j))
#define EAST IDX(i + 1, j, Nb)
#define WEST IDX(i - 1, j, Nb)
#define NORTH IDX(i, j + 1, Nb)
#define SOUTH IDX(i, j - 1, Nb)
#define CENTER IDX(i, j, Nb)

#ifdef USE_CUDA
#include <cuda_runtime.h>

// ===================== DOT PRODUCT KERNELS =====================
// Dot product: sum(u1[i] * u2[i])
__global__ void calculate_dot_product_kernel(real_t *u1, real_t *u2, real_t *dotproduct, real_t *temp, int N, int Nb)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
	if (i <= N && j <= N)
	{
		int linear_idx = (i-1) * N + (j-1);
		temp[linear_idx] = u1[CENTER] * u2[CENTER];
	}
}

// Operator-inverse weighted dot product: sum(u1[i] * u2[i] * CoPinv[i]^2)
__global__ void calculate_opinv_dot_product_kernel(real_t *u1, real_t *u2, real_t *CoPinv, real_t *dotproduct, real_t *temp, int N, int Nb)
{
	int i = blockIdx.y * blockDim.y + threadIdx.y + 1; 
	int j = blockIdx.x * blockDim.x + threadIdx.x + 1; 
	if (i <= N && j <= N)
	{
		int linear_idx = (i-1) * N + (j-1);
		temp[linear_idx] = u1[CENTER] * u2[CENTER] * CoPinv[CENTER] * CoPinv[CENTER];
	}
}

// Residual norm calculation
// Residual norm calculation
__global__ void calculate_residual_norm_kernel(real_t *u, real_t *rhs, real_t *CoE, real_t *CoW, real_t *CoN, real_t *CoS, real_t *CoP, real_t *result, int N, int Nb)
{
    int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
    int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
    if (i <= N && j <= N)
    {
        real_t residual = rhs[CENTER] 
                        - (CoP[CENTER] * u[CENTER]
                         + CoE[CENTER] * u[EAST]
                         + CoW[CENTER] * u[WEST]
                         + CoN[CENTER] * u[NORTH]
                         + CoS[CENTER] * u[SOUTH]);
        
        // Get absolute value of residual
        #ifdef USE_DOUBLE
            result[CENTER] = fabs(residual);
        #else
            result[CENTER] = fabsf(residual);
        #endif
    }
}

// ===================== VECTOR OPERATIONS KERNELS =====================
// Vector scalar addition: result = alpha * phi + beta
__global__ void calculate_vector_scalar_addition_kernel(real_t alpha, real_t *phi_u, real_t beta, real_t *result_u, int N, int Nb)
{
		int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
		int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
		if (i <= N && j <= N)
		{
				result_u[CENTER] = alpha * phi_u[CENTER] + beta;
		}
}

// Element-wise vector product: result[i] = u1[i] * u2[i]
__global__ void calculate_vector_product_kernel(real_t* u1, real_t* u2, real_t* result, int N, int Nb)
{
		int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
		int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
		if (i <= N && j <= N)
		{
				result[CENTER] = u1[CENTER] * u2[CENTER];
		}
}

// Vector linear combination: result = alpha1 * phi1 + alpha2 * phi2
__global__ void calculate_vector_linear_combination_kernel(real_t alpha1, real_t alpha2, real_t *phi1_u, real_t *phi2_u, real_t *result_u, int N, int Nb)
{
		int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
		int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
		if (i <= N && j <= N)
		{
				result_u[CENTER] = alpha1 * phi1_u[CENTER] + alpha2 * phi2_u[CENTER];
		}
}

// Vector addition: result = alpha1 * phi1 + alpha2 * phi2
__global__ void calculate_vector_addition_kernel(real_t alpha1, real_t alpha2, real_t *phi1_u, real_t *phi2_u, real_t *result_u, int N, int Nb)
{
		int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
		int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
		if (i <= N && j <= N)
		{
				result_u[CENTER] = alpha1 * phi1_u[CENTER] + alpha2 * phi2_u[CENTER];
		}
}

// ===================== MATRIX/RESIDUAL KERNELS =====================
// Matrix-vector multiplication: result = A * phi
__global__ void calculate_matrix_vector_kernel(real_t* phi_u, real_t* CoE, real_t* CoW, real_t* CoN, real_t* CoS, real_t* CoP, real_t* result_u, int N, int Nb)
{
		int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
		int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
		if (i <= N && j <= N)
		{
				result_u[CENTER] = CoP[CENTER] * phi_u[CENTER] +
													 CoE[CENTER] * phi_u[EAST] +
													 CoW[CENTER] * phi_u[WEST] +
													 CoN[CENTER] * phi_u[NORTH] +
													 CoS[CENTER] * phi_u[SOUTH];
		}
}


// Residual calculation: result = rhs - (A * u)
__global__ void calculate_residual_kernel(real_t* u_u, real_t* rhs_u, real_t* CoE, real_t* CoW, real_t* CoN, real_t* CoS, real_t* CoP, real_t* result_u, int N, int Nb)
{
		int i = blockIdx.y * blockDim.y + threadIdx.y + 1;
		int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
		if (i <= N && j <= N)
		{
				result_u[CENTER] = rhs_u[CENTER]
												- (CoP[CENTER] * u_u[CENTER]
												 + CoE[CENTER] * u_u[EAST]
												 + CoW[CENTER] * u_u[WEST]
												 + CoN[CENTER] * u_u[NORTH]
												 + CoS[CENTER] * u_u[SOUTH]);
		}
}

// ===================== BOUNDARY CONDITION KERNELS =====================
// East boundary
__global__ void calculate_boundary_values_east_kernel(real_t* phi_u, int N, int Nb, int BC_TYPE, real_t valbc_east)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
	switch(BC_TYPE)
	{
		case BC_PERIODIC:
			if (j <= N) { phi_u[IDX(N + 1, j, Nb)] = phi_u[IDX(1, j, Nb)]; }
			break;
		case BC_NEUMANN:
			if (j <= N) { phi_u[IDX(N + 1, j, Nb)] = phi_u[IDX(N, j, Nb)]; }
			break;
		case BC_DIRICHLET:
			if (j <= N) { phi_u[IDX(N + 1, j, Nb)] = 2.0 * valbc_east - phi_u[IDX(N, j, Nb)]; }
			break;
	}
}

// West boundary
__global__ void calculate_boundary_values_west_kernel(real_t* phi_u, int N, int Nb, int BC_TYPE, real_t valbc_west)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x + 1;
	switch(BC_TYPE)
	{
		case BC_PERIODIC:
			if (j <= N) { phi_u[IDX(0, j, Nb)] = phi_u[IDX(N, j, Nb)]; }
			break;
		case BC_NEUMANN:
			if (j <= N) { phi_u[IDX(0, j, Nb)] = phi_u[IDX(1, j, Nb)]; }
			break;
		case BC_DIRICHLET:
			if (j <= N) { phi_u[IDX(0, j, Nb)] = 2.0 * valbc_west - phi_u[IDX(1, j, Nb)]; }
			break;
	}
}

// North boundary
__global__ void calculate_boundary_values_north_kernel(real_t* phi_u, int N, int Nb, int BC_TYPE, real_t valbc_north)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	switch(BC_TYPE)
	{
		case BC_PERIODIC:
			if (i <= N) { phi_u[IDX(i, N + 1, Nb)] = phi_u[IDX(i, 1, Nb)]; }
			break;
		case BC_NEUMANN:
			if (i <= N) { phi_u[IDX(i, N + 1, Nb)] = phi_u[IDX(i, N, Nb)]; }
			break;
		case BC_DIRICHLET:
			if (i <= N) { phi_u[IDX(i, N + 1, Nb)] = 2.0 * valbc_north - phi_u[IDX(i, N, Nb)]; }
			break;
	}
}

// South boundary
__global__ void calculate_boundary_values_south_kernel(real_t* phi_u, int N, int Nb, int BC_TYPE, real_t valbc_south)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + 1;
	switch(BC_TYPE)
	{
		case BC_PERIODIC:
			if (i <= N) { phi_u[IDX(i, 0, Nb)] = phi_u[IDX(i, N, Nb)]; }
			break;
		case BC_NEUMANN:
			if (i <= N) { phi_u[IDX(i, 0, Nb)] = phi_u[IDX(i, 1, Nb)]; }
			break;
		case BC_DIRICHLET:
			if (i <= N) { phi_u[IDX(i, 0, Nb)] = 2.0 * valbc_south - phi_u[IDX(i, 1, Nb)]; }
			break;
	}
}
#endif // USE_CUDA
