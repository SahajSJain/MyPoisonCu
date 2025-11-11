#include <cfloat>
// SOLVERS/JACOBI/jacobi.cpp
// Jacobi iteration step for solving linear system
// solve as: phi_new = phi_old + omega * Dinv * (rhs - A * phi_old)
// But: residual = rhs - A * phi_old
// Therefore: phi_new = phi_old + omega * Dinv * residual
//     i.e. do Jacobi update in modified richardson form

#include <iostream>
#include <fstream>
#include <cstring>
#include "../../INCLUDE/structs.cuh"
#include "../../CALCULATORS/calculators.cuh"
#include "../../MEMMANAGE/memmanage.cuh"
#include "../../INPUTOUTPUT/inputoutput.h"
#include "jacobi.h"

using std::cout;

void jacobi(Solver *Sol, Setup *setup, real_t *time_total, real_t *time_iteration, int *num_iterations)
{
  memcpy(Sol->phi.u, Sol->phi_initial.u, Sol->phi.Ntotal * sizeof(real_t));
#ifdef USE_ACC
  // For OpenACC: delete old data and create fresh
  copySolverToDeviceACC(Sol);
#endif
  // SAVE TO HISTORY/history.jacobi.<simtype>.csv 
  std::ofstream history("HISTORY/jacobi." SIMTYPE ".csv");
  int iter = 0;
  int outeriter = 0;
  int maxiter = setup->maxiter;                             // number of Jacobi iterations per call
  int inneriter = setup->inneriter;                         // number of smoothing steps
  int maxouteriter = (maxiter + inneriter - 1) / inneriter; // maximum number of outer iterations (rounded up)
  real_t time_start, time_end;
  real_t time_start_2, time_end_2;
  real_t total_time = 0.0;
  real_t current_time = 0.0;
  real_t norm = REAL_MAX;      // initial residual norm to a large number
  real_t omega = setup->omega; // relaxation parameter
  // set phi_old = phi at start of outer iteration
  // we do this by doing: phi_old = 1.0* phi + 0.0* temp
  cout << " =============================================================== \n";
  cout << "                        JACOBI SOLVER (ON HOST)";
  cout << " =============================================================== \n";
  cout << "  Max Iterations: " << maxiter << "\n";
  cout << "  Tolerance:      " << setup->tolerance << "\n";
  cout << "  Inner Iter:     " << inneriter << "\n";
  cout << " =============================================================== \n";

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
      jacobi_step(Sol, omega);
    }
    // Every few iterations calculate residual norm
    norm = 0.0; 
    // calculate_residual_norm(&Sol->phi, &Sol->rhs, &Sol->A, &Sol->residual, &norm);
    calculate_dot_product(&Sol->residual, &Sol->residual, &norm);
    norm = sqrt(norm) / (Sol->N * Sol->N);
    time_end_2 = timer();
    current_time = time_end_2 - time_start_2;
    total_time += current_time;
    printf(" %9d | %14.6e | %13.8f | %14.8f\n",
           outeriter * inneriter, norm, current_time / inneriter, total_time);
    history << outeriter * inneriter << ", " << norm << ", " << current_time / inneriter << ", " << total_time << "\n";
    if (norm < setup->tolerance)
    {
      break; // converged
    }
  }
  cout << "---------------------------------------------------------------------------\n";
  time_end = timer();
  cout << " =============================================================== \n";
  cout << " Total HOST JACOBI Time (s): " << time_end - time_start << "\n";
  cout << " Total HOST JACOBI Time per iteration (s): " << (time_end - time_start) / ((outeriter + 1) * inneriter) << "\n";
  cout << " Total HOST JACOBI Outer Iterations: " << outeriter + 1 << "\n";
  cout << " Total HOST JACOBI Iterations: " << (outeriter + 1) * inneriter << "\n";
  cout << " =============================================================== \n";
  *time_total = time_end - time_start;
  *time_iteration = *time_total / ((outeriter + 1) * inneriter);
  *num_iterations = (outeriter + 1) * inneriter;
#ifdef USE_ACC
  copySolverToHostACC(Sol);
#endif
  history.close();
  // write tecplot file for debugging:
  if (writeTecplotOutput(setup, Sol, "jacobi." SIMTYPE ".dat") != 0)
  {
    printf("Error writing tecplot output jacobi." SIMTYPE ".dat.\n");
  }
}

void jacobi_device(Solver *Sol, Setup *setup, real_t *time_total, real_t *time_iteration, int *num_iterations)
{
  #ifdef USE_CUDA
  memcpy(Sol->phi.u, Sol->phi_initial.u, Sol->phi.Ntotal * sizeof(real_t));
  // Copy to device
  copySolverToDevice(Sol);
  // START!!!
  std::ofstream history("HISTORY/jacobi." SIMTYPE ".device.csv");
  int iter = 0;
  int outeriter = 0;
  int maxiter = setup->maxiter;                             // number of Jacobi iterations per call
  int inneriter = setup->inneriter;                         // number of smoothing steps
  int maxouteriter = (maxiter + inneriter - 1) / inneriter; // maximum number of outer iterations (rounded up)
  real_t time_start, time_end;
  real_t time_start_2, time_end_2;
  real_t total_time = 0.0;
  real_t current_time = 0.0;
  real_t norm = REAL_MAX;      // initial residual norm to a large number
  real_t omega = setup->omega; // relaxation parameter
  int numThreads = Sol->numthreads;
  int numBlocks = Sol->numblocks;
  // set phi_old = phi at start of outer iteration
  // we do this by doing: phi_old = 1.0* phi + 0.0* temp
  cout << " =============================================================== \n";
  cout << "                        JACOBI SOLVER (ON DEVICE)";
  cout << " =============================================================== \n";
  cout << "  Max Iterations: " << maxiter << "\n";
  cout << "  Tolerance:      " << setup->tolerance << "\n";
  cout << "  Inner Iter:     " << inneriter << "\n";
  cout << "  Num Threads:    " << numThreads << "\n";
  cout << "  Num Blocks:     " << numBlocks << "\n";
  cout << " =============================================================== \n";

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
      jacobi_step_device(Sol, omega);
    }
    // Every few iterations calculate residual norm
    norm = 0.0;
    // calculate_residual_norm_device(&Sol->phi, &Sol->rhs, &Sol->A, &Sol->residual, &norm,
    //           numThreads, numBlocks);
    calculate_dot_product_device(&Sol->residual, &Sol->residual, &norm,
                                 numThreads, numBlocks);
    norm = sqrt(norm) / (Sol->N * Sol->N); // normalize by number of points
    time_end_2 = timer();
    current_time = time_end_2 - time_start_2;
    total_time += current_time;
    printf(" %9d | %14.6e | %13.8f | %14.8f\n",
           outeriter * inneriter, norm, current_time / inneriter, total_time);
    history << outeriter * inneriter << ", " << norm << ", " << current_time / inneriter << ", " << total_time << "\n";
    if (norm < setup->tolerance)
    {
      break; // converged
    }
  }
  cout << "---------------------------------------------------------------------------\n";

  // Synchronize device before stopping timer
  cudaDeviceSynchronize();

  time_end = timer();
  cout << " =============================================================== \n";
  cout << " Total DEVICE JACOBI Time (s): " << time_end - time_start << "\n";
  cout << " Total DEVICE JACOBI Time per iteration (s): " << (time_end - time_start) / ((outeriter + 1) * inneriter) << "\n";
  cout << " Total DEVICE JACOBI Outer Iterations: " << outeriter + 1 << "\n";
  cout << " Total DEVICE JACOBI Iterations: " << (outeriter + 1) * inneriter << "\n";
  cout << " =============================================================== \n";
  *time_total = time_end - time_start;
  *time_iteration = *time_total / ((outeriter + 1) * inneriter);
  *num_iterations = (outeriter + 1) * inneriter;
  // Copy results back to host
  copySolverToHost(Sol);
  history.close();
  // write tecplot file for debugging:
  if (writeTecplotOutput(setup, Sol, "jacobi." SIMTYPE ".device.dat") != 0)
  {
    printf("Error writing tecplot output jacobi." SIMTYPE ".device.dat.\n");
  }
  #endif
}
