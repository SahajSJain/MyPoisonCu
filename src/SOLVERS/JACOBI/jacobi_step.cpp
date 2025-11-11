// SOLVERS/JACOBI/jacobi_step.cu
// Jacobi iteration step for solving linear system
// solve as: phi_new = phi_old + omega * Dinv * (rhs - A * phi_old) 
// But: residual = rhs - A * phi_old 
// Therefore: phi_new = phi_old + omega * Dinv * residual 
//     i.e. do Jacobi update in modified richardson form 

#include "../../INCLUDE/structs.cuh"
#include "../../CALCULATORS/calculators.cuh"

void jacobi_step(Solver* Sol, real_t omega) {
    int N = Sol->N;
    int Nb = Sol->Nb;
    Operator* A = &Sol->A;
    Field* rhs = &Sol->rhs;
    Field* temp = &Sol->temp;
    Field* phi_old = &Sol->phi; // use current phi as phi_old  
    // We do this as in the current implementation, we do not need to have separate old and new phi fields.
    Field* phi = &Sol->phi;
    Field* residual = &Sol->residual; 
    
    // 1. Calculate residual: residual = rhs - A * phi_old 
    calculate_residual(phi_old, rhs, A, residual); 
    
    // 2. Multiply residual by Dinv (element-wise), store in temp: temp = Dinv * residual
    real_t* Dinv = A->CoPinv;
    real_t* res_u = residual->u;
    real_t* temp_u = temp->u;
    calculate_vector_product(res_u, Dinv, temp_u, N, Nb); 
    
    // 3. Update phi: phi = phi_old + omega * temp
    calculate_vector_linear_combination(ONE, omega, phi_old, temp, phi); 

    // apply boundary conditions to updated phi
    calculate_boundary_values(phi, Sol->bc_east, Sol->bc_west,  
                              Sol->bc_north, Sol->bc_south,
                              Sol->valbc_east, Sol->valbc_west,
                              Sol->valbc_north, Sol->valbc_south);
}

void jacobi_step_device(Solver* Sol, real_t omega) {
    #ifdef USE_CUDA
    int N = Sol->N;
    int Nb = Sol->Nb;
    Operator* A = &Sol->A;
    Field* rhs = &Sol->rhs;
    Field* temp = &Sol->temp;
    Field* phi_old = &Sol->phi; // use current phi as phi_old
    Field* phi = &Sol->phi;
    Field* residual = &Sol->residual;
    
    int numThreads = Sol->numthreads;
    int numBlocks = Sol->numblocks;
    
    // 1. Calculate residual: residual = rhs - A * phi_old 
    calculate_residual_device(phi_old, rhs, A, residual, numThreads, numBlocks); 
    
    // 2. Multiply residual by Dinv (element-wise), store in temp: temp = Dinv * residual
    // Use DEVICE pointers!
    real_t* Dinv_d = A->CoPinv_d;
    real_t* res_u_d = residual->u_d;
    real_t* temp_u_d = temp->u_d;
    calculate_vector_product_device(res_u_d, Dinv_d, temp_u_d, N, Nb, numThreads, numBlocks); 
    
    // 3. Update phi: phi = phi_old + omega * temp
    calculate_vector_linear_combination_device(ONE, omega, phi_old, temp, phi, numThreads, numBlocks); 

    // apply boundary conditions to updated phi
    calculate_boundary_values_device(phi, Sol->bc_east, Sol->bc_west,  
                                     Sol->bc_north, Sol->bc_south,
                                     Sol->valbc_east, Sol->valbc_west,
                                     Sol->valbc_north, Sol->valbc_south,
                                     numThreads);
    #endif
}
