//bicgstab_step.cpp
  // Preconditioned BiCGSTAB Algorithm
    // 
    // 1. r0 = b - A*x0
    // 2. Choose an arbitrary vector r0_hat such that (r0_hat, r0) != 0, e.g., r0_hat = r0
    // 3. rho0 = (r0_hat, r0)
    // 4. p0 = r0
    // 5. For i = 1, 2, 3, ...
    //    1. y = K2^(-1) * K1^(-1) * p_{i-1}
    //    2. v = A*y
    //    3. alpha = rho_{i-1} / (r0_hat, v)
    //    4. h = x_{i-1} + alpha*y
    //    5. s = r_{i-1} - alpha*v
    //    6. If h is accurate enough then x_i = h and quit
    //    7. z = K2^(-1) * K1^(-1) * s
    //    8. t = A*z
    //    9. omega = (K1^(-1)*t, K1^(-1)*s) / (K1^(-1)*t, K1^(-1)*t)
    //    10. x_i = h + omega*z
    //    11. r_i = s - omega*t
    //    12. If x_i is accurate enough then quit
    //    13. rho_i = (r0_hat, r_i)
    //    14. beta = (rho_i/rho_{i-1}) * (alpha/omega)
    //    15. p_i = r_i + beta*(p_{i-1} - omega*v)
    //
    // This formulation is equivalent to applying unpreconditioned BiCGSTAB to the explicitly preconditioned system
    // A_hat * x_hat = b_hat
    //
    // with A_hat = K1^(-1) * A * K2^(-1), x_hat = K2*x and b_hat = K1^(-1)*b. 
    // In other words, both left- and right-preconditioning are possible with this formulation.
    // only allocate if using bicgstab method

#include "../../INCLUDE/structs.cuh"
#include "../../CALCULATORS/calculators.cuh"

real_t bicgstab_step(Solver &Sol, real_t restol) {
    int N = Sol.N;
    int Nb = Sol.Nb;
    int numThreads = Sol.numthreads;
    int numBlocks = Sol.numblocks;
    
    // 1. y = K2^(-1) * K1^(-1) * p_{i-1}
    // Select appropriate pointers based on backend
    #ifdef USE_CUDA
    real_t* Dinv_ptr = Sol.A.CoPinv_d;
    real_t* p_ptr = Sol.bi_p->u_d;
    real_t* y_ptr = Sol.bi_y->u_d;
    #else
    real_t* Dinv_ptr = Sol.A.CoPinv;
    real_t* p_ptr = Sol.bi_p->u;
    real_t* y_ptr = Sol.bi_y->u;
    #endif
    
    calculate_vector_product(p_ptr, Dinv_ptr, y_ptr, N, Nb, numThreads, numBlocks);
    calculate_boundary_values(*Sol.bi_y, Sol, numThreads);

    // 2. v = A*y
    calculate_matrix_vector(*Sol.bi_y, Sol.A, *Sol.bi_v, numThreads, numBlocks);
    
    // 3. alpha = rho_{i-1} / (r0_hat, v)
    real_t alpha_denominator = 0.0;
    calculate_dot_product(*Sol.bi_r0hat, *Sol.bi_v, alpha_denominator, numThreads, numBlocks);
    real_t alpha = Sol.bis_rho0 / alpha_denominator;
    Sol.bis_alpha = alpha;
    
    // 4. h = x_{i-1} + alpha*y
    calculate_vector_linear_combination(ONE, alpha, Sol.phi, *Sol.bi_y, *Sol.bi_h, numThreads, numBlocks);
    
    // 5. s = r_{i-1} - alpha*v
    calculate_vector_linear_combination(ONE, -alpha, Sol.residual, *Sol.bi_v, *Sol.bi_s, numThreads, numBlocks);
    
    // 6. If h is accurate enough then x_i = h and quit
    real_t norm_h = 0.0;
    calculate_residual_norm(*Sol.bi_h, Sol.rhs, Sol.A, Sol.temp, norm_h, numThreads, numBlocks);
    if(norm_h < restol) {
        calculate_vector_linear_combination(ONE, ZERO, *Sol.bi_h, *Sol.bi_h, Sol.phi, numThreads, numBlocks);
        calculate_boundary_values(Sol.phi, Sol, numThreads);
        return norm_h;
    }
    
    // 7. z = K2^(-1) * K1^(-1) * s
    #ifdef USE_CUDA
    real_t* s_ptr = Sol.bi_s->u_d;
    real_t* z_ptr = Sol.bi_z->u_d;
    #else
    real_t* s_ptr = Sol.bi_s->u;
    real_t* z_ptr = Sol.bi_z->u;
    #endif
    
    calculate_vector_product(s_ptr, Dinv_ptr, z_ptr, N, Nb, numThreads, numBlocks);
    calculate_boundary_values(*Sol.bi_z, Sol, numThreads);
    
    // 8. t = A*z
    calculate_matrix_vector(*Sol.bi_z, Sol.A, *Sol.bi_t, numThreads, numBlocks);
    
    // 9. omega = (K1^(-1)*t, K1^(-1)*s) / (K1^(-1)*t, K1^(-1)*t)
    real_t omega_numerator = 0.0;
    real_t omega_denominator = 0.0;
    calculate_opinv_dot_product(*Sol.bi_t, *Sol.bi_s, Sol.A, omega_numerator, numThreads, numBlocks);
    calculate_opinv_dot_product(*Sol.bi_t, *Sol.bi_t, Sol.A, omega_denominator, numThreads, numBlocks);
    real_t omega = omega_numerator / omega_denominator;
    Sol.bis_omega = omega;
    
    // 10. x_i = h + omega*z
    calculate_vector_linear_combination(ONE, omega, *Sol.bi_h, *Sol.bi_z, Sol.phi, numThreads, numBlocks);
    calculate_boundary_values(Sol.phi, Sol, numThreads);
    
    // 11. r_i = s - omega*t
    calculate_vector_linear_combination(ONE, -omega, *Sol.bi_s, *Sol.bi_t, Sol.residual, numThreads, numBlocks);
    
    // 12. If x_i is accurate enough then quit
    real_t norm_x = 0.0;
    calculate_residual_norm(Sol.phi, Sol.rhs, Sol.A, Sol.temp, norm_x, numThreads, numBlocks);
    if(norm_x < restol) {
        return norm_x;
    }
    
    // 13. rho_i = (r0_hat, r_i)
    real_t rho1 = 0.0;
    calculate_dot_product(*Sol.bi_r0hat, Sol.residual, rho1, numThreads, numBlocks);
    Sol.bis_rho1 = rho1;
    
    // 14. beta = (rho_i/rho_{i-1}) * (alpha/omega)
    real_t beta = (rho1 / Sol.bis_rho0) * (alpha / omega);
    Sol.bis_beta = beta;
    
    // 15. p_i = r_i + beta*(p_{i-1} - omega*v)
    calculate_vector_linear_combination(ONE, -omega, *Sol.bi_p, *Sol.bi_v, Sol.temp, numThreads, numBlocks);
    calculate_vector_linear_combination(ONE, beta, Sol.residual, Sol.temp, *Sol.bi_p, numThreads, numBlocks);
    
    // Update rho0 for next iteration
    Sol.bis_rho0 = rho1;
    return norm_x;
}