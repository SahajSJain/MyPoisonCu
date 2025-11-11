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

real_t bicgstab_step(Solver* Sol, real_t restol) {
    // Function returns residual. 
    // For i = 1, 2, 3, ...
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

    int N = Sol->N;
    int Nb = Sol->Nb;
    Operator* A = &Sol->A;
    Field* temp = &Sol->temp;
    // We do this as in the current implementation, we do not need to have separate old and new phi fields.
    Field* bi_r0 = &Sol->residual; 
    Field* bi_p = &Sol->bi_p;
    Field* bi_v = &Sol->bi_v;
    Field* bi_y = &Sol->bi_y;
    Field* bi_h = &Sol->bi_h;
    Field* bi_s = &Sol->bi_s;
    Field* bi_t = &Sol->bi_t;
    Field* bi_z = &Sol->bi_z;
    Field* bi_x = &Sol->phi; // solution field
    Field* bi_r0hat = &Sol->bi_r0hat; // shadow residual
    // 1. y = K2^(-1) * K1^(-1) * p_{i-1} 
        real_t* Dinv = A->CoPinv; // preconditioner is inverse of diagonal 
        real_t* p_u = bi_p->u;  
        real_t* y_u = bi_y->u; 
        calculate_vector_product(p_u, Dinv, y_u, N, Nb); 
        // apply boundary conditions to updated y 
        calculate_boundary_values(bi_y, Sol->bc_east, Sol->bc_west,  
                              Sol->bc_north, Sol->bc_south,
                              Sol->valbc_east, Sol->valbc_west,
                              Sol->valbc_north, Sol->valbc_south);

    // 2. v = A*y 
        // void calculate_matrix_vector(Field *phi, Operator *op, Field *result)
        calculate_matrix_vector(bi_y, A, bi_v); 
    // 3. alpha = rho_{i-1} / (r0_hat, v) 
        real_t alpha_numerator = Sol->bis_rho0; 
        //calculate_dot_product(Field *u1, Field *u2, real_t* result) 
        real_t alpha_denominator = 0.0;
        calculate_dot_product(bi_r0hat, bi_v, &alpha_denominator);
        real_t alpha = alpha_numerator / alpha_denominator;
        Sol->bis_alpha = alpha;
    // 4. h = x_{i-1} + alpha*y
        //void calculate_vector_linear_combination(real_t alpha1, real_t alpha2, Field *phi1,  Field *phi2, Field *result)
        calculate_vector_linear_combination(ONE, alpha, bi_x, bi_y, bi_h); 
    // 5. s = r_{i-1} - alpha*v
        calculate_vector_linear_combination(ONE, -alpha, bi_r0, bi_v, bi_s);
    // 6. If h is accurate enough then x_i = h and quit 
        // Calculate residual norm for h: 
        // Set temporary residual field to r_h = b - A* h
        Field* bi_hres = &Sol->temp; // temporary field for residual of h 
        real_t norm_h = 0.0;   
        // calculate_residual_norm(Field *u, Field *rhs, Operator *op, Field *result, real_t* norm)
        calculate_residual_norm(bi_h, &Sol->rhs, A, bi_hres, &norm_h);
        if(norm_h < restol) {
            // copy solution from h to x. // Works because h and x are different fields 
            calculate_vector_linear_combination(ONE, ZERO, bi_h, bi_h, bi_x); 
            // apply boundary conditions to updated x
            calculate_boundary_values(bi_x, Sol->bc_east, Sol->bc_west,  
                                  Sol->bc_north, Sol->bc_south,
                                  Sol->valbc_east, Sol->valbc_west,
                                  Sol->valbc_north, Sol->valbc_south);
            return norm_h; // converged
        } 
    // Skipping convergence check for now // Repeatedly checking for convergence here would be inefficient
    // 7. z = K2^(-1) * K1^(-1 ) * s
        real_t* s_u = bi_s->u;  
        real_t* z_u = bi_z->u; 
        calculate_vector_product(s_u, Dinv, z_u, N, Nb); 
        // apply boundary conditions to updated z 
        calculate_boundary_values(bi_z, Sol->bc_east, Sol->bc_west,  
                              Sol->bc_north, Sol->bc_south,
                              Sol->valbc_east, Sol->valbc_west,
                              Sol->valbc_north, Sol->valbc_south);
    // 8. t = A*z
        calculate_matrix_vector(bi_z, A, bi_t);
    // 9. omega = (K1^(-1)*t, K1^(-1)*s) / (K1^(-1)*t, K1^(-1)*t)
        real_t omega_numerator = 0.0;
        real_t omega_denominator = 0.0;
        // void calculate_opinv_dot_product(Field *u1, Field *u2, Operator *op, real_t* result)
        calculate_opinv_dot_product(bi_t, bi_s, A, &omega_numerator);
        calculate_opinv_dot_product(bi_t, bi_t, A, &omega_denominator);
        real_t omega = omega_numerator / omega_denominator;
        Sol->bis_omega = omega;
    // 10. x_i = h + omega*z
        calculate_vector_linear_combination(ONE, omega, bi_h, bi_z, bi_x);
        // apply boundary conditions to updated x
        calculate_boundary_values(bi_x, Sol->bc_east, Sol->bc_west,  
                              Sol->bc_north, Sol->bc_south,
                              Sol->valbc_east, Sol->valbc_west,
                              Sol->valbc_north, Sol->valbc_south);
    // 11. r_i = s - omega*t
        calculate_vector_linear_combination(ONE, -omega, bi_s, bi_t, bi_r0);
    // 12. If x_i is accurate enough then quit 
        Field* bi_xres = &Sol->temp; // temporary field for residual of h 
        real_t norm_x = 0.0;   
        // calculate_residual_norm(Field *u, Field *rhs, Operator *op, Field *result, real_t* norm)
        calculate_residual_norm(bi_x, &Sol->rhs, A, bi_xres, &norm_x);
        if(norm_x < restol) {
            // apply boundary conditions to updated x
            calculate_boundary_values(bi_x, Sol->bc_east, Sol->bc_west,  
                                  Sol->bc_north, Sol->bc_south,
                                  Sol->valbc_east, Sol->valbc_west,
                                  Sol->valbc_north, Sol->valbc_south);
            return norm_x; // converged
        } 

    // 13. rho_i = (r0_hat, r_i)
        real_t rho1 = 0.0;
        calculate_dot_product(bi_r0hat, bi_r0, &rho1);
        Sol->bis_rho1 = rho1; 
    // 14. beta = (rho_i/rho_{i-1}) * (alpha/omega)
        real_t beta = (rho1 / Sol->bis_rho0) * (alpha / omega);
        Sol->bis_beta = beta;
    // 15. p_i = r_i + beta*(p_{i-1} - omega*v)
        // first calculate (p_{i-1} - omega*v) and store in temp
        calculate_vector_linear_combination(ONE, -omega, bi_p, bi_v, temp);
        // now calculate p_i = r_i + beta*temp
        calculate_vector_linear_combination(ONE, beta, bi_r0, temp, bi_p);
    // update rho0 for next iteration
    Sol->bis_rho0 = rho1;
    return norm_x; // not yet converged
    // end of bicgstab step
}

real_t bicgstab_step_device(Solver* Sol, real_t restol) {
    #ifdef USE_CUDA
    // Function returns residual. 
    // Preconditioned BiCGSTAB Algorithm - Device Version
    // For i = 1, 2, 3, ...
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

    int N = Sol->N;
    int Nb = Sol->Nb;
    Operator* A = &Sol->A;
    Field* temp = &Sol->temp;
    // We do this as in the current implementation, we do not need to have separate old and new phi fields.
    Field* bi_r0 = &Sol->residual; 
    Field* bi_p = &Sol->bi_p;
    Field* bi_v = &Sol->bi_v;
    Field* bi_y = &Sol->bi_y;
    Field* bi_h = &Sol->bi_h;
    Field* bi_s = &Sol->bi_s;
    Field* bi_t = &Sol->bi_t;
    Field* bi_z = &Sol->bi_z;
    Field* bi_x = &Sol->phi; // solution field
    Field* bi_r0hat = &Sol->bi_r0hat; // shadow residual
    
    int numThreads = Sol->numthreads;
    int numBlocks = Sol->numblocks;
    
    // 1. y = K2^(-1) * K1^(-1) * p_{i-1} 
    // Preconditioner is inverse of diagonal (CoPinv)
    real_t* Dinv_d = A->CoPinv_d;
    real_t* p_u_d = bi_p->u_d;  
    real_t* y_u_d = bi_y->u_d; 
    calculate_vector_product_device(p_u_d, Dinv_d, y_u_d, N, Nb, numThreads, numBlocks); 
    
    // Apply boundary conditions to updated y 
    calculate_boundary_values_device(bi_y, Sol->bc_east, Sol->bc_west,  
                          Sol->bc_north, Sol->bc_south,
                          Sol->valbc_east, Sol->valbc_west,
                          Sol->valbc_north, Sol->valbc_south,
                          numThreads);

    // 2. v = A*y 
    calculate_matrix_vector_device(bi_y, A, bi_v, numThreads, numBlocks); 
    
    // 3. alpha = rho_{i-1} / (r0_hat, v) 
    real_t alpha_numerator = Sol->bis_rho0; 
    real_t alpha_denominator = 0.0;
    calculate_dot_product_device(bi_r0hat, bi_v, &alpha_denominator, numThreads, numBlocks);
    real_t alpha = alpha_numerator / alpha_denominator;
    Sol->bis_alpha = alpha;
    
    // 4. h = x_{i-1} + alpha*y
    calculate_vector_linear_combination_device(ONE, alpha, bi_x, bi_y, bi_h, numThreads, numBlocks); 
    
    // 5. s = r_{i-1} - alpha*v
    calculate_vector_linear_combination_device(ONE, -alpha, bi_r0, bi_v, bi_s, numThreads, numBlocks);
    
    // 6. If h is accurate enough then x_i = h and quit 
        // Calculate residual norm for h: 
        // Set temporary residual field to r_h = b - A* h
        Field* bi_hres = &Sol->temp; // temporary field for residual of h 
        real_t norm_h = 0.0;   
        // calculate_residual_norm(Field *u, Field *rhs, Operator *op, Field *result, real_t* norm)
        calculate_residual_norm_device(bi_h, &Sol->rhs, A, bi_hres, &norm_h, numThreads, numBlocks);
        if(norm_h < restol) {
            // copy solution from h to x. // Works because h and x are different fields 
            calculate_vector_linear_combination_device(ONE, ZERO, bi_h, bi_h, bi_x, numThreads, numBlocks); 
            // apply boundary conditions to updated x
            calculate_boundary_values_device(bi_x, Sol->bc_east, Sol->bc_west,  
                                  Sol->bc_north, Sol->bc_south,
                                  Sol->valbc_east, Sol->valbc_west,
                                  Sol->valbc_north, Sol->valbc_south,
                                  numThreads);
            return norm_h; // converged
        }
    
    // 7. z = K2^(-1) * K1^(-1) * s
    real_t* s_u_d = bi_s->u_d;  
    real_t* z_u_d = bi_z->u_d; 
    calculate_vector_product_device(s_u_d, Dinv_d, z_u_d, N, Nb, numThreads, numBlocks); 
    
    // Apply boundary conditions to updated z 
    calculate_boundary_values_device(bi_z, Sol->bc_east, Sol->bc_west,  
                          Sol->bc_north, Sol->bc_south,
                          Sol->valbc_east, Sol->valbc_west,
                          Sol->valbc_north, Sol->valbc_south,
                          numThreads);
    
    // 8. t = A*z
    calculate_matrix_vector_device(bi_z, A, bi_t, numThreads, numBlocks);
    
    // 9. omega = (K1^(-1)*t, K1^(-1)*s) / (K1^(-1)*t, K1^(-1)*t)
    real_t omega_numerator = 0.0;
    real_t omega_denominator = 0.0;
    calculate_opinv_dot_product_device(bi_t, bi_s, A, &omega_numerator, numThreads, numBlocks);
    calculate_opinv_dot_product_device(bi_t, bi_t, A, &omega_denominator, numThreads, numBlocks);
    real_t omega = omega_numerator / omega_denominator;
    Sol->bis_omega = omega;
    
    // 10. x_i = h + omega*z
    calculate_vector_linear_combination_device(ONE, omega, bi_h, bi_z, bi_x, numThreads, numBlocks);
    
    // Apply boundary conditions to updated x
    calculate_boundary_values_device(bi_x, Sol->bc_east, Sol->bc_west,  
                          Sol->bc_north, Sol->bc_south,
                          Sol->valbc_east, Sol->valbc_west,
                          Sol->valbc_north, Sol->valbc_south,
                          numThreads);
    
    // 11. r_i = s - omega*t
    calculate_vector_linear_combination_device(ONE, -omega, bi_s, bi_t, bi_r0, numThreads, numBlocks);
    
    // 12. If x_i is accurate enough then quit 
        Field* bi_xres = &Sol->temp; // temporary field for residual of h 
        real_t norm_x = 0.0;   
        // calculate_residual_norm(Field *u, Field *rhs, Operator *op, Field *result, real_t* norm)
        calculate_residual_norm_device(bi_x, &Sol->rhs, A, bi_xres, &norm_x, numThreads, numBlocks);
        if(norm_x < restol) {
            // apply boundary conditions to updated x
            calculate_boundary_values_device(bi_x, Sol->bc_east, Sol->bc_west,  
                                  Sol->bc_north, Sol->bc_south,
                                  Sol->valbc_east, Sol->valbc_west,
                                  Sol->valbc_north, Sol->valbc_south,
                                  numThreads);
            return norm_x; // converged
        } 

    // 13. rho_i = (r0_hat, r_i)
    real_t rho1 = 0.0;
    calculate_dot_product_device(bi_r0hat, bi_r0, &rho1, numThreads, numBlocks);
    Sol->bis_rho1 = rho1; 
    
    // 14. beta = (rho_i/rho_{i-1}) * (alpha/omega)
    real_t beta = (rho1 / Sol->bis_rho0) * (alpha / omega);
    Sol->bis_beta = beta;
    
    // 15. p_i = r_i + beta*(p_{i-1} - omega*v)
    // First calculate (p_{i-1} - omega*v) and store in temp
    calculate_vector_linear_combination_device(ONE, -omega, bi_p, bi_v, temp, numThreads, numBlocks);
    // Now calculate p_i = r_i + beta*temp
    calculate_vector_linear_combination_device(ONE, beta, bi_r0, temp, bi_p, numThreads, numBlocks);
    
    // Update rho0 for next iteration
    Sol->bis_rho0 = rho1;
    return norm_x; // not yet converged
    // End of bicgstab step on device
    #endif
    return ZERO; // if not using CUDA, return zero
}

