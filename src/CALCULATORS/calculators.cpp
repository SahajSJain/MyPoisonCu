// CALCULATORS/calculators.cpp
// Unified interface for calculator operations that switches between host and device implementations

#include "calculators.cuh"
#include "../INCLUDE/structs.cuh"
#include <chrono>
// basic structure: 
//  #ifdef USE_CUDA
//      call device version
//  #else
//      call host version
//  #endif
// this means CUDA+X will result in device versions called.

// Timer function - returns current time in seconds
double timer() {
    using namespace std::chrono;
    static auto start = steady_clock::now();
    auto now = steady_clock::now();
    duration<double> elapsed = now - start;
    return elapsed.count();
}

// ===================== RESIDUAL OPERATIONS =====================
void calculate_residual(Field<real_t> &u, Field<real_t> &rhs, Operator &op, Field<real_t> &result, int numThreads, int numBlocks)
{
#ifdef USE_CUDA
    calculate_residual_device(u, rhs, op, result, numThreads, numBlocks);
#else
    calculate_residual_host(u, rhs, op, result);
#endif
}

void calculate_residual_norm(Field<real_t> &u, Field<real_t> &rhs, Operator &op, Field<real_t> &result, real_t &norm, int numThreads, int numBlocks)
{
#ifdef USE_CUDA
    calculate_residual_norm_device(u, rhs, op, result, norm, numThreads, numBlocks);
#else
    calculate_residual_norm_host(u, rhs, op, result, norm);
#endif
}

// ===================== MATRIX/VECTOR OPERATIONS =====================
void calculate_matrix_vector(Field<real_t> &phi, Operator &op, Field<real_t> &result, int numThreads, int numBlocks)
{
#ifdef USE_CUDA
    calculate_matrix_vector_device(phi, op, result, numThreads, numBlocks);
#else
    calculate_matrix_vector_host(phi, op, result);
#endif
}

// ===================== DOT PRODUCT OPERATIONS =====================
void calculate_dot_product(Field<real_t> &u1, Field<real_t> &u2, real_t &result, 
                          Field<real_t> &temp, int numThreads, int numBlocks)
{
#ifdef USE_CUDA
    calculate_dot_product_device(u1, u2, result, temp, numThreads, numBlocks);
#else
    calculate_dot_product_host(u1, u2, result);
    // Note: temp is not used in host version
#endif
}

void calculate_opinv_dot_product(Field<real_t> &u1, Field<real_t> &u2, Operator &op, 
                                real_t &result, Field<real_t> &temp, 
                                int numThreads, int numBlocks)
{
#ifdef USE_CUDA
    calculate_opinv_dot_product_device(u1, u2, op, result, temp, numThreads, numBlocks);
#else
    calculate_opinv_dot_product_host(u1, u2, op, result);
    // Note: temp is not used in host version
#endif
}

// ===================== VECTOR OPERATIONS =====================
void calculate_vector_scalar_addition(real_t alpha, Field<real_t> &phi, real_t beta, Field<real_t> &result, int numThreads, int numBlocks)
{
#ifdef USE_CUDA
    calculate_vector_scalar_addition_device(alpha, phi, beta, result, numThreads, numBlocks);
#else
    calculate_vector_scalar_addition_host(alpha, phi, beta, result);
#endif
}

void calculate_vector_product(real_t *u1, real_t *u2, real_t *result, int N, int Nb, int numThreads, int numBlocks)
{
#ifdef USE_CUDA
    calculate_vector_product_device(u1, u2, result, N, Nb, numThreads, numBlocks);
#else
    calculate_vector_product_host(u1, u2, result, N, Nb);
#endif
}

void calculate_vector_linear_combination(real_t alpha1, real_t alpha2, Field<real_t> &phi1, Field<real_t> &phi2, Field<real_t> &result, int numThreads, int numBlocks)
{
#ifdef USE_CUDA
    calculate_vector_linear_combination_device(alpha1, alpha2, phi1, phi2, result, numThreads, numBlocks);
#else
    calculate_vector_linear_combination_host(alpha1, alpha2, phi1, phi2, result);
#endif
}

void calculate_vector_addition(real_t alpha1, real_t alpha2, Field<real_t> &phi1, Field<real_t> &phi2, Field<real_t> &result, int numThreads, int numBlocks)
{
#ifdef USE_CUDA
    calculate_vector_addition_device(alpha1, alpha2, phi1, phi2, result, numThreads, numBlocks);
#else
    calculate_vector_addition_host(alpha1, alpha2, phi1, phi2, result);
#endif
}

// ===================== BOUNDARY CONDITIONS =====================
void calculate_boundary_values(Field<real_t> &phi, Solver &solver, int numThreads)
{
#ifdef USE_CUDA
    calculate_boundary_values_device(phi, solver, numThreads);
#else
    calculate_boundary_values_host(phi, solver);
#endif
}