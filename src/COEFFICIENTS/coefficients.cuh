#ifndef INITIALIZE_SOLVER_CUH
#define INITIALIZE_SOLVER_CUH

#include <cuda_runtime.h>
#include <math.h>
#include "../INCLUDE/structs.cuh"

// Macro definitions for indexing
#define IDX(i, j, Nb) ((i) * (Nb) + (j))
#define EAST IDX(i + 1, j, Nb)
#define WEST IDX(i - 1, j, Nb)
#define NORTH IDX(i, j + 1, Nb)
#define SOUTH IDX(i, j - 1, Nb)
#define CENTER IDX(i, j, Nb)

// Mathematical constants (if not defined in structs.cuh)
#ifndef PI
#define PI 3.14159265358979323846f
#endif

// Helper function declarations (implemented in initializeSolver.cu)
inline real_t rhsfieldcalc(real_t x, real_t y);
inline real_t solfieldcalc(real_t x, real_t y);

// Main initialization function
// Clears device memory and copies everything to device
void initializeSolver(Solver *Sol, Setup *setup);

// Initialization type constants
#define INIT_ZERO 0
#define INIT_MANUFACTURED 1
#define INIT_RANDOM 2

// RHS type constants  
#define RHS_ZERO 0
#define RHS_MANUFACTURED 1
#define RHS_RANDOM 2

#endif // INITIALIZE_SOLVER_CUH
