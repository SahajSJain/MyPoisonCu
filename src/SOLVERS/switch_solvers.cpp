#include "switch_solvers.h"
#include "./JACOBI/jacobi.h"
#include "./SRJ/SRJ.h"
#include "./BiCGSTAB/bicgstab.h"
#include "../INCLUDE/structs.cuh"
void switch_solver(Solver *Sol, Setup *setup, real_t *time_total, real_t *time_iteration, int *num_iterations)
{
// If USE_DEVICE==1, run device solver (only if USE_CUDA is defined)
#ifdef USE_CUDA
    switch (setup->method)
    {
    case METHOD_JACOBI:
        jacobi_device(Sol, setup, time_total, time_iteration, num_iterations);
        break;
    case METHOD_SRJ:
        SRJ_device(Sol, setup, time_total, time_iteration, num_iterations);
        break;
    case METHOD_BICGSTAB:
        bicgstab_device(Sol, setup, time_total, time_iteration, num_iterations);
        break;
    default:
        break;
    }
#else
    // If USE_HOST==1, run host solver
    switch (setup->method)
    {
    case METHOD_JACOBI:
        jacobi(Sol, setup, time_total, time_iteration, num_iterations);
        break;
    case METHOD_SRJ:
        SRJ(Sol, setup, time_total, time_iteration, num_iterations);
        break;
    case METHOD_BICGSTAB:
        bicgstab(Sol, setup, time_total, time_iteration, num_iterations);
        break;
    default:
        break;
    }
#endif
}
