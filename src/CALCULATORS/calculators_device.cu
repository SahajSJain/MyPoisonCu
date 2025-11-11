// CALCULATORS/calculators_device.cu
// All device (non-kernel) functions for calculator operations

#include "calculators.cuh"
#include "../INCLUDE/structs.cuh"

// Index macros for 2D grid with boundaries
#define IDX(i, j, Nb) ((i) * (Nb) + (j))
#define EAST IDX(i + 1, j, Nb)
#define WEST IDX(i - 1, j, Nb)
#define NORTH IDX(i, j + 1, Nb)
#define SOUTH IDX(i, j - 1, Nb)
#define CENTER IDX(i, j, Nb)

// ===================== DOT PRODUCT DEVICE FUNCTIONS =====================
void calculate_dot_product_device(Field<real_t> &u1, Field<real_t> &u2, real_t &result,
								  int numThreads, int numBlocks)
{
#ifdef USE_CUDA
  int N = u1.N;
  int Nb = u1.Nb;
  real_t *d_dotproduct;
  cudaMalloc((void**)&d_dotproduct, sizeof(real_t));
  real_t zero = ZERO;
  cudaMemcpy(d_dotproduct, &zero, sizeof(real_t), cudaMemcpyHostToDevice);
  dim3 blockSize(numThreads, numThreads);
  dim3 gridSize(numBlocks, numBlocks);
  calculate_dot_product_kernel<<<gridSize, blockSize>>>(
	  u1.u_d, u2.u_d, d_dotproduct, N, Nb);
  cudaDeviceSynchronize();
  real_t host_result;
  cudaMemcpy(&host_result, d_dotproduct, sizeof(real_t), cudaMemcpyDeviceToHost);
  cudaFree(d_dotproduct);
  result = host_result;
#endif
}

void calculate_opinv_dot_product_device(Field<real_t> &u1, Field<real_t> &u2, Operator &op, real_t &result,
										int numThreads, int numBlocks)
{
#ifdef USE_CUDA
  int N = u1.N;
  int Nb = u1.Nb;
  real_t *d_dotproduct;
  cudaMalloc((void**)&d_dotproduct, sizeof(real_t));
  real_t zero = ZERO;
  cudaMemcpy(d_dotproduct, &zero, sizeof(real_t), cudaMemcpyHostToDevice);
  dim3 blockSize(numThreads, numThreads);
  dim3 gridSize(numBlocks, numBlocks);
  calculate_opinv_dot_product_kernel<<<gridSize, blockSize>>>(
	  u1.u_d, u2.u_d, op.CoPinv_d, d_dotproduct, N, Nb);
  cudaDeviceSynchronize();
  real_t host_result;
  cudaMemcpy(&host_result, d_dotproduct, sizeof(real_t), cudaMemcpyDeviceToHost);
  cudaFree(d_dotproduct);
  result = host_result;
#endif
}

// ===================== VECTOR OPERATIONS DEVICE FUNCTIONS =====================
void calculate_vector_scalar_addition_device(real_t alpha, Field<real_t> &phi, real_t beta, Field<real_t> &result, int numThreads, int numBlocks)
{
#ifdef USE_CUDA
	int N = result.N;
	int Nb = result.Nb;
	dim3 blockSize(numThreads, numThreads);
	dim3 gridSize(numBlocks, numBlocks);
	calculate_vector_scalar_addition_kernel<<<gridSize, blockSize>>>(
		alpha,
		phi.u_d,
		beta,
		result.u_d,
		N,
		Nb
	);
	cudaDeviceSynchronize();
#endif
}

void calculate_vector_product_device(real_t* u1_d, real_t* u2_d, real_t* result_d,
									int N, int Nb, int numThreads, int numBlocks)
{
#ifdef USE_CUDA
	dim3 blockSize(numThreads, numThreads);
	dim3 gridSize(numBlocks, numBlocks);
	calculate_vector_product_kernel<<<gridSize, blockSize>>>(
		u1_d, u2_d, result_d, N, Nb);
	cudaDeviceSynchronize();
#endif
}

void calculate_vector_linear_combination_device(real_t alpha1, real_t alpha2, Field<real_t> &phi1, Field<real_t> &phi2, Field<real_t> &result, int numThreads, int numBlocks)
{
#ifdef USE_CUDA
	int N = result.N;
	int Nb = result.Nb;
	dim3 blockSize(numThreads, numThreads);
	dim3 gridSize(numBlocks, numBlocks);
	calculate_vector_linear_combination_kernel<<<gridSize, blockSize>>>(
		alpha1,
		alpha2,
		phi1.u_d,
		phi2.u_d,
		result.u_d,
		N,
		Nb
	);
	cudaDeviceSynchronize();
#endif
}

void calculate_vector_addition_device(real_t alpha1, real_t alpha2, Field<real_t> &phi1, Field<real_t> &phi2, Field<real_t> &result, int numThreads, int numBlocks)
{
#ifdef USE_CUDA
	int N = result.N;
	int Nb = result.Nb;
	dim3 blockSize(numThreads, numThreads);
	dim3 gridSize(numBlocks, numBlocks);
	calculate_vector_addition_kernel<<<gridSize, blockSize>>>(
		alpha1,
		alpha2,
		phi1.u_d,
		phi2.u_d,
		result.u_d,
		N,
		Nb
	);
	cudaDeviceSynchronize();
#endif
}

// ===================== MATRIX/RESIDUAL DEVICE FUNCTIONS =====================
void calculate_matrix_vector_device(Field<real_t> &phi, Operator &op, Field<real_t> &result, int numThreads, int numBlocks)
{
#ifdef USE_CUDA
	int N = op.N;
	int Nb = op.Nb;
	dim3 blockSize(numThreads, numThreads);
	dim3 gridSize(numBlocks, numBlocks);
	calculate_matrix_vector_kernel<<<gridSize, blockSize>>>(
		phi.u_d,
		op.CoE_d,
		op.CoW_d,
		op.CoN_d,
		op.CoS_d,
		op.CoP_d,
		result.u_d,
		N,
		Nb
	);
	cudaDeviceSynchronize();
#endif
}

void calculate_residual_norm_device(Field<real_t> &u, Field<real_t> &rhs, Operator &op, Field<real_t> &result, real_t &norm, int numThreads, int numBlocks)
{
#ifdef USE_CUDA
  int N = op.N;
  int Nb = op.Nb;
  real_t *d_norm;
  cudaMalloc((void**)&d_norm, sizeof(real_t));
  real_t zero = ZERO;
  cudaMemcpy(d_norm, &zero, sizeof(real_t), cudaMemcpyHostToDevice);
  dim3 blockSize(numThreads, numThreads);
  dim3 gridSize(numBlocks, numBlocks);
  calculate_residual_norm_kernel<<<gridSize, blockSize>>>(
	  u.u_d, rhs.u_d, op.CoE_d, op.CoW_d, op.CoN_d, op.CoS_d, op.CoP_d, result.u_d, d_norm, N, Nb);
  cudaDeviceSynchronize();
  real_t host_norm;
  cudaMemcpy(&host_norm, d_norm, sizeof(real_t), cudaMemcpyDeviceToHost);
  cudaFree(d_norm);
  norm = host_norm / (N * N);
#endif
}

void calculate_residual_device(Field<real_t> &u, Field<real_t> &rhs, Operator &op, Field<real_t> &result, int numThreads, int numBlocks)
{
#ifdef USE_CUDA
	int N = op.N;
	int Nb = op.Nb;
	dim3 blockSize(numThreads, numThreads);
	dim3 gridSize(numBlocks, numBlocks);
	calculate_residual_kernel<<<gridSize, blockSize>>>(
		u.u_d, rhs.u_d,
		op.CoE_d, op.CoW_d, op.CoN_d, op.CoS_d, op.CoP_d,
		result.u_d, N, Nb
	);
	cudaDeviceSynchronize();
#endif
}

// ===================== BOUNDARY CONDITION DEVICE FUNCTIONS =====================
void calculate_boundary_values_device(Field<real_t> &phi, int bc_east, int bc_west, int bc_north, int bc_south, real_t valbc_east, real_t valbc_west, real_t valbc_north, real_t valbc_south, int numThreads)
{
#ifdef USE_CUDA
  int N = phi.N;
  int Nb = phi.Nb;
  int numBlocks = (N + numThreads - 1) / numThreads;
  calculate_boundary_values_east_kernel<<<numBlocks, numThreads>>>(
	  phi.u_d, N, Nb, bc_east, valbc_east);
  calculate_boundary_values_west_kernel<<<numBlocks, numThreads>>>(
	  phi.u_d, N, Nb, bc_west, valbc_west);
  calculate_boundary_values_north_kernel<<<numBlocks, numThreads>>>(
	  phi.u_d, N, Nb, bc_north, valbc_north);
  calculate_boundary_values_south_kernel<<<numBlocks, numThreads>>>(
	  phi.u_d, N, Nb, bc_south, valbc_south);
  cudaDeviceSynchronize();
#endif
}

void calculate_boundary_values_device(Field<real_t> &phi, Solver &s, int numThreads)
{
#ifdef USE_CUDA
  int N = phi.N;
  int Nb = phi.Nb;
  int numBlocks = (N + numThreads - 1) / numThreads;
  calculate_boundary_values_east_kernel<<<numBlocks, numThreads>>>(
	  phi.u_d, N, Nb, s.bc_east, s.valbc_east);
  calculate_boundary_values_west_kernel<<<numBlocks, numThreads>>>(
	  phi.u_d, N, Nb, s.bc_west, s.valbc_west);
  calculate_boundary_values_north_kernel<<<numBlocks, numThreads>>>(
	  phi.u_d, N, Nb, s.bc_north, s.valbc_north);
  calculate_boundary_values_south_kernel<<<numBlocks, numThreads>>>(
	  phi.u_d, N, Nb, s.bc_south, s.valbc_south);
  cudaDeviceSynchronize();
#endif
}