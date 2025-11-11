#ifndef SWITCH_SOLVERS_H
#define SWITCH_SOLVERS_H

#include "../INCLUDE/structs.cuh"

void switch_solver(Solver* Sol, Setup* setup, real_t* time_total, real_t* time_iteration, int* num_iterations);

#endif // SWITCH_SOLVERS_H
