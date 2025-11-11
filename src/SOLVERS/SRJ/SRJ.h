// SOLVERS/SRJ/SRJ.h
// Header for SRJ (Successive Richardson-Jacobi) solver functions

#ifndef SRJ_H
#define SRJ_H

#include "../../INCLUDE/structs.cuh"

// SRJ solver - host version
// Returns: total time, time per iteration, number of iterations
void SRJ(Solver* Sol, Setup* setup, real_t* time_total, real_t* time_iteration, int* num_iterations);

#endif // SRJ_H
