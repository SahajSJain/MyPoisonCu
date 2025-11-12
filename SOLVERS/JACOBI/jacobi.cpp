// jacobi.cpp
#include <cfloat>
// SOLVERS/JACOBI/jacobi.cpp
// Jacobi solver using modified Richardson iteration

#include <iostream>
#include <fstream>
#include <cstring>
#include "../../INCLUDE/structs.cuh"
#include "../../CALCULATORS/calculators.cuh"
#include "../../INPUTOUTPUT/inputoutput.h"
#include "jacobi.h"

using std::cout;

void jacobi(Solver &Sol, Setup *setup, real_t *time_total, real_t *time_iteration, int *num_iterations)
{
  // Initialize phi from phi_initial
  Sol.phi = Sol.phi_initial;
  
  // Upload to device if using CUDA/ACC
  Sol.upload();
  
  // SAVE TO HISTORY/jacobi.<simtype>.csv 
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
  real_t current_total_time = 0.0;
  int numThreads = Sol.numthreads;
  int numBlocks = Sol.numblocks;
  
  cout << " =============================================================== \n";
  #ifdef USE_CUDA
  cout << "                        JACOBI SOLVER (ON DEVICE)\n";
  cout << "  Num Threads:    " << numThreads << "\n";
  cout << "  Num Blocks:     " << numBlocks << "\n";
  #else
  cout << "                        JACOBI SOLVER (ON HOST)\n";
  #endif
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
  history << "Iteration, Residual_Norm, Total_Time\n"; 
  
  for (outeriter = 1; outeriter <= maxouteriter; outeriter++)
  {
    time_start_2 = timer();
    for (iter = 1; iter <= inneriter; iter++)
    {
      jacobi_step(Sol, omega);
      
      // Calculate residual norm after each iteration
      norm = 0.0;
      calculate_dot_product(Sol.residual, Sol.residual, norm, numThreads, numBlocks);
      norm = sqrt(norm) / (Sol.N * Sol.N);
      
      // Update total time
      current_total_time = timer() - time_start;
      
      // Write to history file
      history << (outeriter - 1) * inneriter + iter << ", " << norm
              << ", " << current_total_time << "\n";
    }
    
    time_end_2 = timer();
    current_time = time_end_2 - time_start_2;
    total_time += current_time;
    
    printf(" %9d | %14.6e | %13.8f | %14.8f\n",
           outeriter * inneriter, norm, current_time / inneriter, total_time);
    
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
  cout << " Total JACOBI Time (s): " << time_end - time_start << "\n";
  cout << " Total JACOBI Time per iteration (s): " << (time_end - time_start) / (outeriter * inneriter) << "\n";
  cout << " Total JACOBI Outer Iterations: " << outeriter << "\n";
  cout << " Total JACOBI Iterations: " << outeriter * inneriter << "\n";
  cout << " =============================================================== \n";
  
  *time_total = time_end - time_start;
  *time_iteration = *time_total / (outeriter * inneriter);
  *num_iterations = outeriter * inneriter;
  
  // Download from device if using CUDA/ACC
  Sol.download();
  
  history.close();
  
  // write tecplot file for debugging:
  if (writeTecplotOutput(setup, Sol, "jacobi." SIMTYPE ".dat") != 0)
  {
    printf("Error writing tecplot output jacobi." SIMTYPE ".dat.\n");
  }
}