// CALCULATORS/calculators.cuh
// Header file for all calculator operations

#ifndef CALCULATORS_CUH
#define CALCULATORS_CUH

#include "../INCLUDE/structs.cuh"


// Timer function - returns current time in seconds
double timer();

// ===================== HOST INTERFACE =====================
// All host/device interface functions use Field<real_t> & and Operator &
void calculate_residual(Field<real_t> &u, Field<real_t> &rhs, Operator &op, Field<real_t> &result, int numThreads, int numBlocks);
void calculate_residual_norm(Field<real_t> &u, Field<real_t> &rhs, Operator &op, Field<real_t> &result, real_t &norm, int numThreads, int numBlocks);
void calculate_matrix_vector(Field<real_t> &phi, Operator &op, Field<real_t> &result, int numThreads, int numBlocks);
// Note: device implementations use a temporary Field for intermediate reductions
void calculate_dot_product(Field<real_t> &u1, Field<real_t> &u2, real_t &result, Field<real_t> &temp, int numThreads, int numBlocks);
void calculate_opinv_dot_product(Field<real_t> &u1, Field<real_t> &u2, Operator &op, real_t &result, Field<real_t> &temp, int numThreads, int numBlocks);
void calculate_vector_scalar_addition(real_t alpha, Field<real_t> &phi, real_t beta, Field<real_t> &result, int numThreads, int numBlocks);
void calculate_vector_product(real_t *u1, real_t *u2, real_t *result, int N, int Nb, int numThreads, int numBlocks);
void calculate_vector_linear_combination(real_t alpha1, real_t alpha2, Field<real_t> &phi1, Field<real_t> &phi2, Field<real_t> &result, int numThreads, int numBlocks);
void calculate_vector_addition(real_t alpha1, real_t alpha2, Field<real_t> &phi1, Field<real_t> &phi2, Field<real_t> &result, int numThreads, int numBlocks);
void calculate_boundary_values(Field<real_t> &phi, Solver &solver, int numThreads);

// ===================== HOST-ONLY (for host implementations) =====================
void calculate_residual_host(Field<real_t> &u, Field<real_t> &rhs, Operator &op, Field<real_t> &result);
void calculate_residual_norm_host(Field<real_t> &u, Field<real_t> &rhs, Operator &op, Field<real_t> &result, real_t &norm);
void calculate_matrix_vector_host(Field<real_t> &phi, Operator &op, Field<real_t> &result);
void calculate_dot_product_host(Field<real_t> &u1, Field<real_t> &u2, real_t &result);
void calculate_opinv_dot_product_host(Field<real_t> &u1, Field<real_t> &u2, Operator &op, real_t &result);
void calculate_vector_scalar_addition_host(real_t alpha, Field<real_t> &phi, real_t beta, Field<real_t> &result);
void calculate_vector_product_host(real_t *u1, real_t *u2, real_t *result, int N, int Nb);
void calculate_vector_linear_combination_host(real_t alpha1, real_t alpha2, Field<real_t> &phi1, Field<real_t> &phi2, Field<real_t> &result);
void calculate_vector_addition_host(real_t alpha1, real_t alpha2, Field<real_t> &phi1, Field<real_t> &phi2, Field<real_t> &result);
void calculate_boundary_values_host(Field<real_t> &phi, Solver &solver);

#ifdef USE_CUDA
// ===================== DEVICE INTERFACE (CUDA only) =====================
void calculate_residual_device(Field<real_t> &u, Field<real_t> &rhs, Operator &op, Field<real_t> &result, int numThreads, int numBlocks);
void calculate_residual_norm_device(Field<real_t> &u, Field<real_t> &rhs, Operator &op, Field<real_t> &result, real_t &norm, int numThreads, int numBlocks);
void calculate_matrix_vector_device(Field<real_t> &phi, Operator &op, Field<real_t> &result, int numThreads, int numBlocks);
void calculate_dot_product_device(Field<real_t> &u1, Field<real_t> &u2, real_t &result, Field<real_t> &temp, int numThreads, int numBlocks);
void calculate_opinv_dot_product_device(Field<real_t> &u1, Field<real_t> &u2, Operator &op, real_t &result, Field<real_t> &temp, int numThreads, int numBlocks);
void calculate_vector_scalar_addition_device(real_t alpha, Field<real_t> &phi, real_t beta, Field<real_t> &result, int numThreads, int numBlocks);
void calculate_vector_product_device(real_t *u1_d, real_t *u2_d, real_t *result_d, int N, int Nb, int numThreads, int numBlocks);
void calculate_vector_linear_combination_device(real_t alpha1, real_t alpha2, Field<real_t> &phi1, Field<real_t> &phi2, Field<real_t> &result, int numThreads, int numBlocks);
void calculate_vector_addition_device(real_t alpha1, real_t alpha2, Field<real_t> &phi1, Field<real_t> &phi2, Field<real_t> &result, int numThreads, int numBlocks);
void calculate_boundary_values_device(Field<real_t> &phi, int bc_east, int bc_west, int bc_north, int bc_south, real_t valbc_east, real_t valbc_west, real_t valbc_north, real_t valbc_south, int numThreads);
void calculate_boundary_values_device(Field<real_t> &phi, Solver &s, int numThreads);

// ===================== CUDA KERNELS (from calculators_kernels.cu) =====================
extern "C" {
__global__ void calculate_dot_product_kernel(real_t *u1, real_t *u2, real_t *dotproduct, real_t *temp, int N, int Nb);
__global__ void calculate_opinv_dot_product_kernel(real_t *u1, real_t *u2, real_t *CoPinv, real_t *dotproduct, real_t *temp, int N, int Nb);
__global__ void calculate_vector_scalar_addition_kernel(real_t alpha, real_t *phi_u, real_t beta, real_t *result_u, int N, int Nb);
__global__ void calculate_vector_product_kernel(real_t* u1, real_t* u2, real_t* result, int N, int Nb);
__global__ void calculate_vector_linear_combination_kernel(real_t alpha1, real_t alpha2, real_t *phi1_u, real_t *phi2_u, real_t *result_u, int N, int Nb);
__global__ void calculate_vector_addition_kernel(real_t alpha1, real_t alpha2, real_t *phi1_u, real_t *phi2_u, real_t *result_u, int N, int Nb);
__global__ void calculate_matrix_vector_kernel(real_t* phi_u, real_t* CoE, real_t* CoW, real_t* CoN, real_t* CoS, real_t* CoP, real_t* result_u, int N, int Nb);
__global__ void calculate_residual_norm_kernel(real_t *u, real_t *rhs, real_t *CoE, real_t *CoW, real_t *CoN, real_t *CoS, real_t *CoP, real_t *result, real_t *norm, int N, int Nb);
__global__ void calculate_residual_kernel(real_t* u_u, real_t* rhs_u, real_t* CoE, real_t* CoW, real_t* CoN, real_t* CoS, real_t* CoP, real_t* result_u, int N, int Nb);
__global__ void calculate_boundary_values_east_kernel(real_t* phi_u, int N, int Nb, int BC_TYPE, real_t valbc_east);
__global__ void calculate_boundary_values_west_kernel(real_t* phi_u, int N, int Nb, int BC_TYPE, real_t valbc_west);
__global__ void calculate_boundary_values_north_kernel(real_t* phi_u, int N, int Nb, int BC_TYPE, real_t valbc_north);
__global__ void calculate_boundary_values_south_kernel(real_t* phi_u, int N, int Nb, int BC_TYPE, real_t valbc_south);
}

#endif // USE_CUDA

#endif // CALCULATORS_CUH
