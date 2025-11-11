// CALCULATORS/calculators.cuh
// Header file for all calculator operations

#ifndef CALCULATORS_CUH
#define CALCULATORS_CUH

#include "../INCLUDE/structs.cuh"

// Function declarations

// Calculate matrix-vector product result = A * phi
// where A is represented by the Operator coefficients
void calculate_matrix_vector(Field* phi, Operator* op, Field* result);

// ...existing code...

// Calculate residual: result = rhs - A * u
// where A is represented by the Operator coefficients
void calculate_residual(Field* u, Field* rhs, Operator* op, Field* result);

// ...existing code...

// Calculate residual and its L1 norm: result = rhs - A * u
// Result is returned via reference in norm parameter
void calculate_residual_norm(Field* u, Field* rhs, Operator* op, 
                             Field* result, real_t* norm);

// ...existing code...

// Apply boundary conditions to a field (host version)
void calculate_boundary_values(Field *phi, int bc_east, int bc_west, 
                             int bc_north, int bc_south, real_t valbc_east, 
                             real_t valbc_west, real_t valbc_north, real_t valbc_south);

// ...existing code...

// Calculate vector linear combination: result = alpha1 * phi1 + alpha2 * phi2 (host version)
void calculate_vector_linear_combination(real_t alpha1, real_t alpha2, Field *phi1, 
                                        Field *phi2, Field *result);

// ...existing code...

// Calculate dot product: result = sum(u1[i] * u2[i]) (host version)
void calculate_dot_product(Field *u1, Field *u2, real_t* result);

// ...existing code...

// Calculate operator-inverse weighted dot product: result = sum(u1[i] * u2[i] * CoPinv[i]^2) (host version)
void calculate_opinv_dot_product(Field *u1, Field *u2, Operator *op, real_t* result);

// ...existing code...

// Calculate element-wise vector product: result[i] = u1[i] * u2[i] (host version)
// Uses raw pointers instead of Field structs for flexibility
void calculate_vector_product(real_t *u1, real_t *u2, real_t *result, int N, int Nb);

// ...existing code...

// Calculate vector scalar addition: result = alpha * phi + beta (host version)
void calculate_vector_scalar_addition(real_t alpha, Field *phi, real_t beta, Field *result);

// ...existing code...

// Timer function - returns current time in seconds
double timer();

#ifdef USE_CUDA
// Device function declarations (CUDA only)
void calculate_matrix_vector_device(Field* phi, Operator* op, Field* result, 
                                     int numThreads, int numBlocks);
void calculate_residual_device(Field* u, Field* rhs, Operator* op, 
                                Field* result, int numThreads, int numBlocks);
void calculate_residual_norm_device(Field* u, Field* rhs, Operator* op, 
                                     Field* result, real_t* norm,
                                     int numThreads, int numBlocks);
void calculate_boundary_values_device(Field *phi, int bc_east, int bc_west, 
                                     int bc_north, int bc_south, 
                                     real_t valbc_east, real_t valbc_west,
                                     real_t valbc_north, real_t valbc_south,
                                     int numThreads);
void calculate_vector_linear_combination_device(real_t alpha1, real_t alpha2, Field *phi1, 
                                               Field *phi2, Field *result, 
                                               int numThreads, int numBlocks);
void calculate_dot_product_device(Field *u1, Field *u2, real_t* result,
                                  int numThreads, int numBlocks);
void calculate_opinv_dot_product_device(Field *u1, Field *u2, Operator *op, real_t* result,
                                        int numThreads, int numBlocks);
void calculate_vector_product_device(real_t* u1_d, real_t* u2_d, real_t* result_d,
                                    int N, int Nb, int numThreads, int numBlocks);
void calculate_vector_scalar_addition_device(real_t alpha, Field *phi, real_t beta, 
                                             Field *result, int numThreads, int numBlocks);
#endif // USE_CUDA

#endif // CALCULATORS_CUH
