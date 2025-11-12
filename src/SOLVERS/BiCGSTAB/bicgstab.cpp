// SOLVERS/BiCGSTAB/bicgstab.cpp
// BiCGSTAB iteration step for solving linear system
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

#include <iostream>
#include <fstream>
#include <cstring>
#include "../../INCLUDE/structs.cuh"
#include "../../CALCULATORS/calculators.cuh"
#include "../../INPUTOUTPUT/inputoutput.h"
#include "bicgstab.h"

using std::cout;

void bicgstab(Solver &Sol, Setup *setup, real_t *time_total, real_t *time_iteration, int *num_iterations)
{
  // Initialize phi from phi_initial
  Sol.phi = Sol.phi_initial;
  
  // Upload to device if using CUDA/ACC
  Sol.upload();
  
  // SAVE TO HISTORY/bicgstab.<simtype>.csv 
  std::ofstream history("HISTORY/bicgstab." SIMTYPE ".csv");
  int iter = 0;
  int outeriter = 0;
  int maxiter = setup->maxiter;                             // number of BiCGSTAB iterations per call
  int inneriter = setup->inneriter;                         // number of smoothing steps
  int maxouteriter = (maxiter + inneriter - 1) / inneriter; // maximum number of outer iterations (rounded up)
  real_t time_start, time_end;
  real_t time_start_2, time_end_2;
  real_t total_time = 0.0;
  real_t current_time = 0.0;
  real_t norm = REAL_MAX;      // initial residual norm to a large number
  int numThreads = Sol.numthreads;
  int numBlocks = Sol.numblocks;
  int total_iterations = 0;
  
  cout << " =============================================================== \n";
  #ifdef USE_CUDA
  cout << "                        BICGSTAB SOLVER (ON DEVICE)\n";
  cout << "  Num Threads:    " << numThreads << "\n";
  cout << "  Num Blocks:     " << numBlocks << "\n";
  #else
  cout << "                        BICGSTAB SOLVER (ON HOST)\n";
  #endif
  cout << " =============================================================== \n";
  cout << "  Max Iterations: " << maxiter << "\n";
  cout << "  Tolerance:      " << setup->tolerance << "\n";
  cout << "  Inner Iter:     " << inneriter << "\n";
  cout << " =============================================================== \n";
  
  // Preprocessing bicgstab: 
  // 1. r0 = b - A*x0
  calculate_residual(Sol.phi, Sol.rhs, Sol.A, Sol.residual, numThreads, numBlocks);  
  
  // 2. Choose an arbitrary vector r0_hat such that (r0_hat, r0) != 0, e.g., r0_hat = r0
  calculate_vector_linear_combination(ONE, ZERO, Sol.residual, Sol.temp, *Sol.bi_r0hat, numThreads, numBlocks); 
  
  // 3. rho0 = (r0_hat, r0)
  real_t rho0 = 0.0; 
  calculate_dot_product(*Sol.bi_r0hat, Sol.residual, rho0, Sol.temp, numThreads, numBlocks); 
  Sol.bis_rho0 = rho0; 
  Sol.bis_rho1 = rho0;
  
  // 4. p0 = r0
  calculate_vector_linear_combination(ONE, ZERO, Sol.residual, Sol.temp, *Sol.bi_p, numThreads, numBlocks); 
  
  // print a header file:
  time_start = timer();
  cout << "---------------------------------------------------------------------------\n";
  cout << " Iteration | Residual Norm  | Time/iter (s) | Total Time (s)\n";
  cout << "---------------------------------------------------------------------------\n";
  history << "Iteration, Residual_Norm, Total_Time\n"; 
  
  for (outeriter = 1; outeriter <= maxouteriter; outeriter++)
  {
    time_start_2 = timer();
    
    // Perform inneriter iterations without norm calculation
    for (iter = 1; iter <= inneriter; iter++)
    {
      bicgstab_step(Sol);
      total_iterations++;
    }
    
    // Calculate residual norm after inneriter iterations
    norm = 0.0;
    calculate_dot_product(Sol.residual, Sol.residual, norm, Sol.temp, numThreads, numBlocks);
    norm = sqrt(norm) / (Sol.N * Sol.N);
    
    time_end_2 = timer();
    current_time = time_end_2 - time_start_2;
    total_time += current_time;
    
    // Update total time for history
    real_t current_total_time = timer() - time_start;
    
    // Write to history file and print to stdout
    history << total_iterations << ", " << norm
            << ", " << current_total_time << "\n";
    
    printf(" %9d | %14.6e | %13.8f | %14.8f\n",
           total_iterations, norm, current_time / inneriter, total_time);
    
    if (norm < setup->tolerance)
    { 
      break; // converged
    } 
  }
  cout << "---------------------------------------------------------------------------\n";

  // Synchronize device before stopping timer if using CUDA
  #ifdef USE_CUDA
  cudaDeviceSynchronize();
  #endif

  time_end = timer();
  cout << " =============================================================== \n";
  cout << " Total BICGSTAB Time (s): " << time_end - time_start << "\n";
  cout << " Total BICGSTAB Time per iteration (s): " << (time_end - time_start) / total_iterations << "\n";
  cout << " Total BICGSTAB Outer Iterations: " << outeriter << "\n";
  cout << " Total BICGSTAB Iterations: " << total_iterations << "\n";
  cout << " =============================================================== \n";
  
  *time_total = time_end - time_start;
  *time_iteration = *time_total / total_iterations;
  *num_iterations = total_iterations;
  
  // Download from device if using CUDA/ACC
  Sol.download();
  
  history.close();
  
  // write tecplot file for debugging:
  if (writeTecplotOutput(setup, Sol, "bicgstab." SIMTYPE ".dat") != 0)
  {
    printf("Error writing tecplot output bicgstab." SIMTYPE ".dat.\n");
  }
}