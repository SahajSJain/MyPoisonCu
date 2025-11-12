// SOLVERS/BiCGSTAB/bicgstab.h
// Header for BiCGSTAB solver functions

#ifndef BICGSTAB_H
#define BICGSTAB_H

#include "../../INCLUDE/structs.cuh"

// BiCGSTAB solver - host version
// Returns: total time, time per iteration, number of iterations
void bicgstab(Solver &Sol, Setup *setup, real_t *time_total, real_t *time_iteration, int *num_iterations);
real_t bicgstab_step(Solver &Sol, real_t restol);

#endif // BICGSTAB_H
