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
#ifdef _OPENMP
#include <omp.h>
#endif
#include <iostream>
#include "INCLUDE/structs.cuh"
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
    //                CHECK OPENMP CONFIGURATION
    // ========================================================================
    #ifdef USE_OMP
    #ifdef _OPENMP
    printf("=============================================================\n");
    printf("OpenMP Configuration:\n");
    printf("  OpenMP Version: %d\n", _OPENMP);
    printf("  Max threads available: %d\n", omp_get_max_threads());
    printf("  Number of processors: %d\n", omp_get_num_procs());
    
    // Get and display thread affinity info
    #pragma omp parallel
    {
        #pragma omp master
        {
            printf("  Number of threads in use: %d\n", omp_get_num_threads());
            printf("  Thread affinity: ");
            if (omp_get_proc_bind() == omp_proc_bind_false)
                printf("false (threads can move between processors)\n");
            else if (omp_get_proc_bind() == omp_proc_bind_true)
                printf("true (threads bound to processors)\n");
            else if (omp_get_proc_bind() == omp_proc_bind_master)
                printf("master (threads bound to same place as master)\n");
            else if (omp_get_proc_bind() == omp_proc_bind_close)
                printf("close (threads bound close to master)\n");
            else if (omp_get_proc_bind() == omp_proc_bind_spread)
                printf("spread (threads spread across processors)\n");
        }
    }
    
    // Try to get CPU info (Linux-specific)
    #ifdef __linux__
    FILE* cpuinfo = fopen("/proc/cpuinfo", "r");
    if (cpuinfo != NULL) {
        char line[256];
        bool found_model = false;
        while (fgets(line, sizeof(line), cpuinfo)) {
            if (strstr(line, "model name") && !found_model) {
                char* model = strchr(line, ':');
                if (model) {
                    model += 2; // Skip ": "
                    model[strcspn(model, "\n")] = 0; // Remove newline
                    printf("  CPU Model: %s\n", model);
                    found_model = true;
                }
            }
        }
        fclose(cpuinfo);
    }
    #endif
    
    printf("=============================================================\n\n");
    #else
    printf("WARNING: USE_OMP is defined but _OPENMP is not!\n");
    printf("         OpenMP may not be properly enabled.\n\n");
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
    // CREATE AND INITIALIZE SOLVER
    // ========================================================================
    // Extract values from setup
    int N = setup.N;
    int Nf = 0;  // Not used for non-multigrid methods
    int Nc = 0;  // Not used for non-multigrid methods
    int level = 0;
    real_t Lxs = setup.Lx_start;
    real_t Lxe = setup.Lx_end;
    real_t Lys = setup.Ly_start;
    real_t Lye = setup.Ly_end;
    
    // Create solver using constructor
    cout << "Creating solver...\n";
    Solver Sol(N, Nf, Nc, level, Lxs, Lxe, Lys, Lye, setup.method);
    cout << "Solver created successfully.\n";
    
    // ========================================================================
    // INITIALIZE SOLVER 
    // ========================================================================
    // Initialize operators and fields on host (also copies boundary conditions from setup)
    cout << "Initializing solver...\n";
    initializeSolver(&Sol, &setup);     
    cout << "Solver initialized successfully.\n";
    
    // ========================================================================
    // CALL THE ACTUAL SOLVER 
    // ========================================================================
    real_t time_total = 0.0;
    real_t time_iteration = 0.0;
    int num_iterations = 0;
    switch_solver(Sol, &setup, &time_total, &time_iteration, &num_iterations);
    
    // ========================================================================
    // Print summary
    // ========================================================================
    cout<<"-------------------------------------------------------------\n";
    cout<<"Solver Summary:\n";
    cout<<"  Method Used: ";
    if (setup.method == METHOD_JACOBI) {    
        cout<<"Jacobi Method\n";
    } else if (setup.method == METHOD_SRJ) {
        cout<<"SRJ Method\n";
    } else if (setup.method == METHOD_BICGSTAB) {
        cout<<"BiCGStab Method\n";
    } else {
        cout<<"Unknown Method\n";
    }
    
    // print computation mode based on preprocessor flags
    cout<<"  Computation Mode: ";
    #ifdef USE_CUDA
    cout<<"CUDA (GPU)\n";
    #elif defined(USE_ACC)
    cout<<"OpenACC\n";
    #elif defined(USE_OMP)
    cout<<"OpenMP (CPU)\n";
    #else
    cout<<"Serial (CPU)\n";
    #endif
    
    cout<<"-------------------------------------------------------------\n";
    cout<<"  Total Time Taken: "<<time_total<<" seconds\n";
    cout<<"  Time per Iteration: "<<time_iteration<<" seconds\n";
    cout<<"  Number of Iterations: "<<num_iterations<<"\n";
    cout<<"-------------------------------------------------------------\n";
    
    // ========================================================================
    // CLEANUP AND EXIT
    // ========================================================================
    // Solver destructor will handle all cleanup automatically
    printf("Done.\n");
    return 0;
}