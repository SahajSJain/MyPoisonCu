#include <cstdio>
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>

// Simple driver for SRJ solver comparison
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _OPENACC
#include <openacc.h>
#endif
#include <iostream>
#include "INCLUDE/structs.cuh"
#include "MEMMANAGE/memmanage.cuh"
#include "COEFFICIENTS/coefficients.cuh"
#include "INPUTOUTPUT/inputoutput.h"
#include "CALCULATORS/calculators.cuh"
#include "SOLVERS/switch_solvers.h"
using namespace std; // for cout should never use it but I dont care
int main(int argc, char** argv) {
    // ========================================================================
    //                CHECK OPENACC GPU AVAILABILITY
    // ========================================================================
    // Check OpenACC GPU availability
    #ifdef USE_ACC
    #ifdef _OPENACC
    int num_devices = acc_get_num_devices(acc_device_nvidia);
    if (num_devices > 0) {
        acc_device_t dev_type = acc_get_device_type();
        int dev_num = acc_get_device_num(dev_type);
        printf("=============================================================\n");
        printf("OpenACC GPU Detection:\n");
        printf("  Number of GPUs available: %d\n", num_devices);
        printf("  Device type: %s\n", (dev_type == acc_device_nvidia) ? "NVIDIA GPU" : "Unknown");
        printf("  Using GPU device: %d\n", dev_num);
        printf("=============================================================\n\n");
    } else {
        printf("WARNING: OpenACC is enabled but NO GPU devices detected!\n");
        printf("         Code will run on CPU (host fallback)\n\n");
    }
    #else
    printf("WARNING: USE_ACC is defined but _OPENACC is not!\n");
    printf("         OpenACC runtime may not be properly initialized.\n\n");
    #endif
    #endif
    // ========================================================================
    // Create setup struct and read from input.dat
    // ======================================================================== 
    Setup setup;
    if (readInput(&setup, "input.dat") != 0) {
        printf("Error reading input file. Exiting.\n");
        return 1;
    }
    cout << "Input file read successfully.\n";
    // ========================================================================
    // ALLOCATE AND INITIALIZE SOLVER
    // ========================================================================
    // Create solver struct and zero-initialize pointers
    cout << "Creating solver struct...\n";
    Solver Sol;
    memset(&Sol, 0, sizeof(Solver));
    // Extract values from setup
    int N = setup.N;
    int Nf = 0;
    int Nc = 0;
    int level = 0;
    real_t Lxs = setup.Lx_start;
    real_t Lxe = setup.Lx_end;
    real_t Lys = setup.Ly_start;
    real_t Lye = setup.Ly_end;
    // Allocate solver - use method from input 
    cout << "Allocating solver...\n";
    allocateSolver(&Sol, N, Nf, Nc, level, Lxs, Lxe, Lys, Lye, setup.method);
    cout << "Solver allocated successfully.\n";
    // ========================================================================
    // INITIALIZE SOLVER 
    // ========================================================================
    // Initialize operators and fields on host (also copies boundary conditions from setup)
    // Note: initializeSolver internally clears device memory and copies everything to device
    cout << "Initializing solver...\n";
    initializeSolver(&Sol, &setup);     
    cout << "Solver initialized successfully.\n";
    // ========================================================================
    // CALL THE ACTUAL SOLVER 
    // ========================================================================
    real_t time_total = 0.0;
    real_t time_iteration = 0.0;
    int num_iterations = 0;
    switch_solver(&Sol, &setup, &time_total, &time_iteration, &num_iterations);
    // ========================================================================
    // Print summary
    // ========================================================================
    cout<<"-------------------------------------------------------------\n";
    cout<<"Solver Summary:\n";
    cout<<"  Method Used: ";
    if (setup.method == 1) {    
        cout<<"SRJ Method\n";
    } else if (setup.method == 2) {
        cout<<"Multigrid Method\n";
    } else if (setup.method == 3) {
        cout<<"BiCGStab Method\n";
    } else {
        cout<<"Unknown Method\n";
    }
    // print if using device or host or both
    cout<<"  Computation Mode: ";
    if (setup.USE_DEVICE == 1 && setup.USE_HOST == 0) {
        cout<<"Device (GPU) Only\n";
    } else if (setup.USE_DEVICE == 0 && setup.USE_HOST == 1) {
        cout<<"Host (CPU) Only\n";
    } else if (setup.USE_DEVICE == 1 && setup.USE_HOST == 1) {
        cout<<"Hybrid Device (GPU) + Host (CPU)\n";
    } else {
        cout<<"No Computation Mode Selected!\n";
    }
    cout<<"-------------------------------------------------------------\n";

    cout<<"  Total Time Taken: "<<time_total<<" seconds\n";
    cout<<"  Time per Iteration: "<<time_iteration<<" seconds\n";
    cout<<"  Number of Iterations: "<<num_iterations<<"\n";
    cout<<"-------------------------------------------------------------\n";
    // ========================================================================
    // CLEANUP AND EXIT
    // ========================================================================
    // Delete device data before deallocating host
    #ifdef USE_ACC
    deleteSolverFromDeviceACC(&Sol);
    #endif
    
    // Cleanup
    deallocateSolver(&Sol);
    clearAllDeviceMemory();
    printf("Done.\n");
    return 0;
}
