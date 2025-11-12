// SOLVERS/JACOBI/jacobi_step.cu
// Jacobi iteration step for solving linear system
// solve as: phi_new = phi_old + omega * Dinv * (rhs - A * phi_old) 
// But: residual = rhs - A * phi_old 
// Therefore: phi_new = phi_old + omega * Dinv * residual 
//     i.e. do Jacobi update in modified richardson form 

#include "../../INCLUDE/structs.cuh"
#include "../../CALCULATORS/calculators.cuh"

void jacobi_step(Solver &Sol, real_t omega) {
    int N = Sol.N;
    int Nb = Sol.Nb;
    int numThreads = Sol.numthreads;
    int numBlocks = Sol.numblocks;
    
    // 1. Calculate residual: residual = rhs - A * phi_old 
    calculate_residual(Sol.phi, Sol.rhs, Sol.A, Sol.residual, numThreads, numBlocks); 
    
    // 2. Multiply residual by Dinv (element-wise), store in temp: temp = Dinv * residual
    // Select appropriate pointers based on backend
    #ifdef USE_CUDA
    real_t* Dinv_ptr = Sol.A.CoPinv_d;
    real_t* res_ptr = Sol.residual.u_d;
    real_t* temp_ptr = Sol.temp.u_d;
    #else
    real_t* Dinv_ptr = Sol.A.CoPinv;
    real_t* res_ptr = Sol.residual.u;
    real_t* temp_ptr = Sol.temp.u;
    #endif
    
    calculate_vector_product(res_ptr, Dinv_ptr, temp_ptr, N, Nb, numThreads, numBlocks); 
    
    // 3. Update phi: phi = phi_old + omega * temp
    calculate_vector_linear_combination(ONE, omega, Sol.phi, Sol.temp, Sol.phi, numThreads, numBlocks); 

    // apply boundary conditions to updated phi
    calculate_boundary_values(Sol.phi, Sol, numThreads);
}