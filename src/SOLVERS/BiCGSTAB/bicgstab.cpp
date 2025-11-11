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
#include "../../MEMMANAGE/memmanage.cuh"
#include "../../INPUTOUTPUT/inputoutput.h"
#include "bicgstab.h"

using std::cout;

void bicgstab(Solver *Sol, Setup *setup, real_t *time_total, real_t *time_iteration, int *num_iterations)
{
  memcpy(Sol->phi.u, Sol->phi_initial.u, Sol->phi.Ntotal * sizeof(real_t));
#ifdef USE_ACC
  // For OpenACC: delete old data and create fresh
  copySolverToDeviceACC(Sol);
#endif
  // SAVE TO HISTORY/history.bicgstab.<simtype>.csv 
  std::ofstream history("HISTORY/bicgstab." SIMTYPE ".csv");
  int iter = 0;
  int outeriter = 0;
  int maxiter = setup->maxiter;                             // number of Jacobi iterations per call
  int inneriter = setup->inneriter;                         // number of smoothing steps
  cout << "  Inner Iter:     " << inneriter << "\n";
  int maxouteriter = (maxiter + inneriter - 1) / inneriter; // maximum number of outer iterations (rounded up)
  real_t time_start, time_end;
  real_t time_start_2, time_end_2;
  real_t total_time = 0.0;
  real_t current_time = 0.0;
  real_t norm = REAL_MAX;      // initial residual norm to a large number
  // set phi_old = phi at start of outer iteration
  // we do this by doing: phi_old = 1.0* phi + 0.0* temp
  cout << " =============================================================== \n";
  cout << "                        BICGSTAB SOLVER (ON HOST)" << "\n";
  cout << " =============================================================== \n";
  cout << "  Max Iterations: " << maxiter << "\n";
  cout << "  Tolerance:      " << setup->tolerance << "\n";
  cout << "  Inner Iter:     " << inneriter << "\n";
  cout << " =============================================================== \n";
  // Preprocessing bicgstab: 
  // 1. r0 = b - A*x0
  // 2. Choose an arbitrary vector r0_hat such that (r0_hat, r0) != 0, e.g., r0_hat = r0
  // 3. rho0 = (r0_hat, r0)
  // 4. p0 = r0
  // 5. For i = 1, 2, 3, ...
  Field* bi_r0 = &Sol->residual; 
  Field* phi   = &Sol->phi;
  Field* rhs   = &Sol->rhs;
  Operator* A  = &Sol->A;
  Field* residual = &Sol->residual; 
  Field* temp     = &Sol->temp;
  Field* bi_r0hat = &Sol->bi_r0hat; // shadow residual
  // 1. r0 = b - A*x0
  // Calculate initial residual: r0 = b - A*x0 
  calculate_residual(phi, rhs, A, residual);  
  // 2. Choose an arbitrary vector r0_hat such that (r0_hat, r0) != 0, e.g., r0_hat = r0
  // copy residual to bi_r0hat. 
  // for this we can use linear_combination: bi_r0hat = 1.0* residual + 0.0* temp 
  //void calculate_vector_linear_combination(real_t alpha1, real_t alpha2, Field *phi1,  Field *phi2, Field *result)
  calculate_vector_linear_combination(ONE, ZERO, residual, temp, bi_r0hat); 
  // 3. rho0 = (r0_hat, r0)
  real_t rho0 = 0.0; 
  calculate_dot_product(bi_r0hat, bi_r0, &rho0); 
  Sol->bis_rho0 = rho0; 
  Sol->bis_rho1 = rho0;
  // 4. p0 = r0
  // for this we can use linear_combination: p0 = 1.0* residual + 0.0* temp 
  calculate_vector_linear_combination(ONE, ZERO, residual, temp, &Sol->bi_p); 
  // print a header file:
  time_start = timer();
  cout << "---------------------------------------------------------------------------\n";
  cout << " Iteration | Residual Norm  | Time/iter (s) | Total Time (s)\n";
  cout << "---------------------------------------------------------------------------\n";
  history << "Iteration, Residual_Norm, Time_per_iter, Total_Time\n"; 
  for (outeriter = 1; outeriter < maxouteriter+1; outeriter++)
  {
    time_start_2 = timer();
    for (iter = 1; iter < inneriter+1; iter++)
    {
      norm = bicgstab_step(Sol, setup->tolerance);
      if (norm < setup->tolerance   )
      { 
        break; // converged
      }
    }
    time_end_2 = timer();
    current_time = time_end_2 - time_start_2;
    total_time += current_time;
    printf(" %9d | %14.6e | %13.8f | %14.8f\n",
           (outeriter) * inneriter, norm, current_time / inneriter, total_time);
    history << (outeriter)  * inneriter << ", " << norm << ", " << current_time / inneriter << ", " << total_time << "\n";
    if (norm < setup->tolerance)
    { 
      break; // converged
    } 
  }
  cout << "---------------------------------------------------------------------------\n";
  time_end = timer();
  cout << " =============================================================== \n";
  cout << " Total HOST BICGSTAB Time (s): " << time_end - time_start << "\n";
  cout << " Total HOST BICGSTAB Time per iteration (s): " << (time_end - time_start) / ((outeriter + 1) * inneriter) << "\n";
  cout << " Total HOST BICGSTAB Outer Iterations: " << outeriter + 1 << "\n";
  cout << " Total HOST BICGSTAB Iterations: " << (outeriter + 1) * inneriter << "\n";
  cout << " =============================================================== \n";
  *time_total = time_end - time_start;
  *time_iteration = *time_total / ((outeriter + 1) * inneriter);
  *num_iterations = (outeriter + 1) * inneriter;
#ifdef USE_ACC
  copySolverToHostACC(Sol);
#endif
  history.close();
  // write tecplot file for debugging:
  if (writeTecplotOutput(setup, Sol, "bicgstab." SIMTYPE ".dat") != 0)
  {
    printf("Error writing tecplot output bicgstab." SIMTYPE ".dat.\n");
  }
}

void bicgstab_device(Solver *Sol, Setup *setup, real_t *time_total, real_t *time_iteration, int *num_iterations)
{
  #ifdef USE_CUDA
  memcpy(Sol->phi.u, Sol->phi_initial.u, Sol->phi.Ntotal * sizeof(real_t));
  // Copy to device
  copySolverToDevice(Sol);
  // START!!!
  std::ofstream history("HISTORY/bicgstab." SIMTYPE ".device.csv");
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
  int numThreads = Sol->numthreads;
  int numBlocks = Sol->numblocks;
  cout << " =============================================================== \n";
  cout << "                        BICGSTAB SOLVER (ON DEVICE)" << "\n";
  cout << " =============================================================== \n";
  cout << "  Max Iterations: " << maxiter << "\n";
  cout << "  Tolerance:      " << setup->tolerance << "\n";
  cout << "  Inner Iter:     " << inneriter << "\n";
  cout << "  Num Threads:    " << numThreads << "\n";
  cout << "  Num Blocks:     " << numBlocks << "\n";
  cout << " =============================================================== \n";
  // Preprocessing bicgstab on device: 
  Field* bi_r0 = &Sol->residual; 
  Field* phi   = &Sol->phi;
  Field* rhs   = &Sol->rhs;
  Operator* A  = &Sol->A;
  Field* residual = &Sol->residual; 
  Field* temp     = &Sol->temp;
  Field* bi_r0hat = &Sol->bi_r0hat; // shadow residual
  // 1. r0 = b - A*x0
  calculate_residual_device(phi, rhs, A, residual, numThreads, numBlocks);  
  // 2. Choose an arbitrary vector r0_hat such that (r0_hat, r0) != 0, e.g., r0_hat = r0
  calculate_vector_linear_combination_device(ONE, ZERO, residual, temp, bi_r0hat, numThreads, numBlocks); 
  // 3. rho0 = (r0_hat, r0)
  real_t rho0 = 0.0; 
  calculate_dot_product_device(bi_r0hat, bi_r0, &rho0, numThreads, numBlocks); 
  Sol->bis_rho0 = rho0; 
  Sol->bis_rho1 = rho0;
  // 4. p0 = r0
  calculate_vector_linear_combination_device(ONE, ZERO, residual, temp, &Sol->bi_p, numThreads, numBlocks); 
  // print a header file:
  time_start = timer();
  cout << "---------------------------------------------------------------------------\n";
  cout << " Iteration | Residual Norm  | Time/iter (s) | Total Time (s)\n";
  cout << "---------------------------------------------------------------------------\n";
  history << "Iteration, Residual_Norm, Time_per_iter, Total_Time\n";
  for (outeriter = 1; outeriter < maxouteriter+1; outeriter++)
  {
    time_start_2 = timer();
    for (iter = 1; iter < inneriter+1; iter++)
    {
      norm = bicgstab_step_device(Sol, setup->tolerance);
      if (norm < setup->tolerance )
      { 
        break; // converged
      }

    }
    time_end_2 = timer();
    current_time = time_end_2 - time_start_2;
    total_time += current_time;
    printf(" %9d | %14.6e | %13.8f | %14.8f\n",
           (outeriter) * inneriter, norm, current_time / inneriter, total_time);
    history << (outeriter)  * inneriter << ", " << norm << ", " << current_time / inneriter << ", " << total_time << "\n";
    if (norm < setup->tolerance )
    {
      break; // converged
    }
  }
  cout << "---------------------------------------------------------------------------\n";

  // Synchronize device before stopping timer
  cudaDeviceSynchronize();

  time_end = timer();
  cout << " =============================================================== \n";
  cout << " Total DEVICE BICGSTAB Time (s): " << time_end - time_start << "\n";
  cout << " Total DEVICE BICGSTAB Time per iteration (s): " << (time_end - time_start) / ((outeriter + 1) * inneriter) << "\n";
  cout << " Total DEVICE BICGSTAB Outer Iterations: " << outeriter + 1 << "\n";
  cout << " Total DEVICE BICGSTAB Iterations: " << (outeriter + 1) * inneriter << "\n";
  cout << " =============================================================== \n";
  *time_total = time_end - time_start;
  *time_iteration = *time_total / ((outeriter + 1) * inneriter);
  *num_iterations = (outeriter + 1) * inneriter;
  // Copy results back to host
  copySolverToHost(Sol);
  history.close();
  // write tecplot file for debugging:
  if (writeTecplotOutput(setup, Sol, "bicgstab." SIMTYPE ".device.dat") != 0)
  {
    printf("Error writing tecplot output bicgstab." SIMTYPE ".device.dat.\n");
  }
  #endif
}
