// SRJ.cpp: Implementation of the SRJ solver
// THANKFULLY WE WILL BE USING jacobi_step.cu FOR SRJ STEPS

#include <iostream>
#include <fstream>
#include <cstring>
#include "../../INCLUDE/structs.cuh"
#include "../../CALCULATORS/calculators.cuh"
#include "../../INPUTOUTPUT/inputoutput.h"
#include "../JACOBI/jacobi.h"
#include "SRJ.h"

using std::cout;

void SRJ(Solver &Sol, Setup *setup, real_t *time_total, real_t *time_iteration, int *num_iterations)
{
  // Initialize phi from phi_initial
  Sol.phi = Sol.phi_initial;

  // Upload to device if using CUDA/ACC
  Sol.upload();

  // SAVE TO HISTORY/SRJ.<simtype>.csv
  std::ofstream history("HISTORY/SRJ." SIMTYPE ".csv");
  int iter = 0;
  int outeriter = 0;
  int maxiter = setup->maxiter;                             // number of SRJ iterations per call
  int inneriter = setup->SRJsize;                           // number of smoothing steps = number of SRJ stages
  int maxouteriter = (maxiter + inneriter - 1) / inneriter; // maximum number of outer iterations (rounded up)
  real_t *SRJomega = setup->SRJomega;                       // array of SRJ relaxation parameters
  real_t time_start, time_end;
  real_t time_start_2, time_end_2;
  real_t total_time = 0.0;
  real_t current_time = 0.0;
  real_t norm = REAL_MAX; // initial residual norm to a large number
  int numThreads = Sol.numthreads;
  int numBlocks = Sol.numblocks;
  int total_iterations = 0;

  cout << " =============================================================== \n";
#ifdef USE_CUDA
  cout << "                        SRJ SOLVER (ON DEVICE)\n";
  cout << "  Num Threads:    " << numThreads << "\n";
  cout << "  Num Blocks:     " << numBlocks << "\n";
#else
  cout << "                        SRJ SOLVER (ON HOST)\n";
#endif
  cout << " =============================================================== \n";
  cout << "  Max Iterations: " << maxiter << "\n";
  cout << "  Tolerance:      " << setup->tolerance << "\n";
  cout << "  Inner Iter:     " << inneriter << " (SRJ stages)\n";
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
    
    // Perform inneriter iterations without norm calculation
    for (iter = 1; iter <= inneriter; iter++)
    {
      jacobi_step(Sol, SRJomega[iter - 1]);
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
  cout << " Total SRJ Time (s): " << time_end - time_start << "\n";
  cout << " Total SRJ Time per iteration (s): " << (time_end - time_start) / total_iterations << "\n";
  cout << " Total SRJ Outer Iterations: " << outeriter << "\n";
  cout << " Total SRJ Iterations: " << total_iterations << "\n";
  cout << " =============================================================== \n";

  *time_total = time_end - time_start;
  *time_iteration = *time_total / total_iterations;
  *num_iterations = total_iterations;

  // Download from device if using CUDA/ACC
  Sol.download();

  history.close();

  // write tecplot file for debugging:
  if (writeTecplotOutput(setup, Sol, "SRJ." SIMTYPE ".dat") != 0)
  {
    printf("Error writing tecplot output SRJ." SIMTYPE ".dat.\n");
  }
}