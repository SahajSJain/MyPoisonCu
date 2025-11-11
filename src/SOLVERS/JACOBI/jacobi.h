// SOLVERS/JACOBI/jacobi.h
// Header for Jacobi solver functions

#ifndef JACOBI_H
#define JACOBI_H

#include "../../INCLUDE/structs.cuh"

// Jacobi solver - host version
// Returns: total time, time per iteration, number of iterations
void jacobi(Solver* Sol, Setup* setup, real_t* time_total, real_t* time_iteration, int* num_iterations);
// Single Jacobi iteration - host version
void jacobi_step(Solver* Sol, real_t omega);
#endif // JACOBI_H
