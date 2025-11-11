// SOLVERS/BiCGSTAB/bicgstab.h
// Header for BiCGSTAB solver functions

#ifndef BICGSTAB_H
#define BICGSTAB_H

#include "../../INCLUDE/structs.cuh"

// BiCGSTAB solver - host version
// Returns: total time, time per iteration, number of iterations
void bicgstab(Solver* Sol, Setup* setup, real_t* time_total, real_t* time_iteration, int* num_iterations);


// Single BiCGSTAB iteration - host version
// Returns: residual norm
real_t bicgstab_step(Solver* Sol, real_t restol);
#ifdef USE_CUDA
// Single BiCGSTAB iteration - device version
// Returns: residual norm
real_t bicgstab_step_device(Solver* Sol, real_t restol);
// BiCGSTAB solver - device version  
// Returns: total time, time per iteration, number of iterations
void bicgstab_device(Solver* Sol, Setup* setup, real_t* time_total, real_t* time_iteration, int* num_iterations);
#endif
#endif // BICGSTAB_H
