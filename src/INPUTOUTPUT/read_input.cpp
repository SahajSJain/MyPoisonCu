// INPUTOUTPUT/read_input.cpp
// Reads configuration from input.dat file and populates Setup struct

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include "inputoutput.h"
#include "../INCLUDE/structs.cuh"

// Helper function to create directory if it doesn't exist
static void make_directory(const char* dirname) {
  struct stat st = {0};
  if (stat(dirname, &st) == -1) {
    mkdir(dirname, 0755);
  }
}

int readInput(Setup* setup, const char* filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open input file '" << filename << "'\n";
        return -1;
    }
    
    std::string line;
    
    // Initialize setup to zero
    std::memset(setup, 0, sizeof(Setup));
    // Skip header lines (3 lines)
    std::getline(file, line); // ===============================================================================
    std::getline(file, line); //                     INPUT FILE FOR MyPoisonousCult
    std::getline(file, line); // ===============================================================================
    std::getline(file, line); // -------------------- GRID CONFIGURATION ------------------------------------
    std::getline(file, line); // N
    file >> setup->N;
    std::getline(file, line); // consume rest of line
    std::getline(file, line); // Lx_s Lx_e Ly_s Ly_e
    file >> setup->Lx_start >> setup->Lx_end >> setup->Ly_start >> setup->Ly_end;
    std::getline(file, line); // consume rest of line
    std::getline(file, line); // -------------------- SOLVER CONFIGURATION ----------------------------------
    std::getline(file, line); // METHOD ErrorCheck omega
    file >> setup->method >> setup->ErrorCheck >> setup->omega;
    std::getline(file, line); // consume rest of line
    
    std::getline(file, line); // maxiter tolerance inneriter
    file >> setup->maxiter >> setup->tolerance >> setup->inneriter;
    std::getline(file, line); // consume rest of line
    std::getline(file, line); // -------------------- FIELD INITIALIZATION ----------------------------------
    std::getline(file, line); // INITIALIZATION_TYPE
    file >> setup->INITIALIZATION_TYPE;
    std::getline(file, line); // consume rest of line
    std::getline(file, line); // RHS_TYPE
    file >> setup->RHS_TYPE;
    std::getline(file, line); // consume rest of line
    std::getline(file, line); // -------------------- BOUNDARY CONDITIONS -----------------------------------
    std::getline(file, line); // bc_east bcd_east valbc_east
    file >> setup->bc_east >> setup->bcd_east >> setup->valbc_east;
    std::getline(file, line); // consume rest of line
    std::getline(file, line); // bc_west bcd_west valbc_west
    file >> setup->bc_west >> setup->bcd_west >> setup->valbc_west;
    std::getline(file, line); // consume rest of line
    std::getline(file, line); // bc_north bcd_north valbc_north
    file >> setup->bc_north >> setup->bcd_north >> setup->valbc_north;
    std::getline(file, line); // consume rest of line
    std::getline(file, line); // bc_south bcd_south valbc_south
    file >> setup->bc_south >> setup->bcd_south >> setup->valbc_south;
    std::getline(file, line); // consume rest of line
    
    file.close();
        
    // Print everything after reading
    std::cout << " ================================================================== \n";
    std::cout << "                          MY POISON CULT \n";
    std::cout << " ================================================================== \n";
    std::cout << "           A CUDA-BASED SOLVER FOR 2D POISSON EQUATION \n";
    std::cout << "                                                   - SAHAJ JAIN     \n";
    std::cout << " ================================================================== \n\n";
    // THIS IS DEFAULT BEHAVIOR
    
    // Print preprocessor flags
    std::cout << "Compiler flags:\n";
    #ifdef USE_OMP
        std::cout << "  USE_OMP: ON\n";
    #else
        std::cout << "  USE_OMP: OFF\n";
    #endif
    
    #ifdef USE_ACC
        // Only for OpenACC will we have USE_DEVICE option 
        std::cout << "  USE_ACC: ON\n";
    #else
        std::cout << "  USE_ACC: OFF\n";
    #endif
    
    #ifdef USE_DOUBLE
        std::cout << "  USE_DOUBLE: ON (double precision)\n";
        std::cout << "  real_t size: " << sizeof(real_t) << " bytes\n";
    #else
        std::cout << "  USE_DOUBLE: OFF (single precision)\n";
        std::cout << "  real_t size: " << sizeof(real_t) << " bytes\n";
    #endif
    std::cout << "\n";

    std::cout << "Reading input from input.dat...\n";
    std::cout << "  N = " << setup->N << "\n";
    std::cout << "  Domain: x=[" << setup->Lx_start << ", " 
              << setup->Lx_end << "], y=[" << setup->Ly_start 
              << ", " << setup->Ly_end << "]\n";
    real_t dx = (setup->Lx_end - setup->Lx_start) / (setup->N);
    real_t dy = (setup->Ly_end - setup->Ly_start) / (setup->N);
    real_t idx2 = 1.0f / (dx * dx);
    real_t idy2 = 1.0f / (dy * dy);
    std::cout << "  Grid spacing and coefficients:\n";
    std::cout << "    dx = " << dx << ", dy = " << dy << "\n"; 
    std::cout << "    idx2 = " << idx2 << ", idy2 = " << idy2 << "\n"; 
    std::cout << "    -(2*idx2 + 2*idy2) = " 
              << -2.0f * (idx2 + idy2) << "\n"; 
    std::cout << "\n";
    std::cout << "\n";

    std::cout << "  Method = " << setup->method 
              << ", ErrorCheck = " << setup->ErrorCheck 
              << ", omega = " << setup->omega << "\n";
    std::cout << "  maxiter = " << setup->maxiter 
              << ", tolerance = " << setup->tolerance 
              << ", inneriter = " << setup->inneriter << "\n";
    std::cout << "  Initialization type = " 
              << setup->INITIALIZATION_TYPE << "\n";
    std::cout << "  RHS type = " << setup->RHS_TYPE << "\n";
    std::cout << "  BC East: type=" << setup->bc_east 
              << ", specifier=" << setup->bcd_east 
              << ", value=" << setup->valbc_east << "\n";
    std::cout << "  BC West: type=" << setup->bc_west 
              << ", specifier=" << setup->bcd_west 
              << ", value=" << setup->valbc_west << "\n";
    std::cout << "  BC North: type=" << setup->bc_north 
              << ", specifier=" << setup->bcd_north 
              << ", value=" << setup->valbc_north << "\n";
    std::cout << "  BC South: type=" << setup->bc_south 
              << ", specifier=" << setup->bcd_south 
              << ", value=" << setup->valbc_south << "\n";
    std::cout << "Input file read successfully.\n\n";
    
    // Validate periodic boundary conditions
    // If periodic BC is set on one side, both sides must be periodic
    bool periodic_x = (setup->bc_east == BC_PERIODIC) || (setup->bc_west == BC_PERIODIC);
    bool periodic_y = (setup->bc_north == BC_PERIODIC) || (setup->bc_south == BC_PERIODIC);
    
    if (periodic_x && (setup->bc_east != BC_PERIODIC || setup->bc_west != BC_PERIODIC)) {
        std::cerr << "ERROR: Periodic BC in x-direction requires BOTH east and west to be periodic!\n";
        std::cerr << "       bc_east = " << setup->bc_east << ", bc_west = " << setup->bc_west << "\n";
        return -1;
    }
    
    if (periodic_y && (setup->bc_north != BC_PERIODIC || setup->bc_south != BC_PERIODIC)) {
        std::cerr << "ERROR: Periodic BC in y-direction requires BOTH north and south to be periodic!\n";
        std::cerr << "       bc_north = " << setup->bc_north << ", bc_south = " << setup->bc_south << "\n";
        return -1;
    }
    
    if (periodic_x) {
        std::cout << "  Periodic BCs enabled in x-direction (east-west)\n";
    }
    if (periodic_y) {
        std::cout << "  Periodic BCs enabled in y-direction (north-south)\n";
    }
    std::cout << "\n";
    // Create TECOUT directory if it doesn't exist
    make_directory("TECOUT");
    // Create HISTORY directory if it doesn't exist
    make_directory("HISTORY");
    
    // -----------------------------------------------------------
    //               CHECK IF SRJ.dat exists. 
    // -----------------------------------------------------------
    // If it exists, read SRJ parameters and populate setup->SRJactive, setup->SRJsize, setup->SRJomega 
    // If Method is SRJ and SRJactive is not set, throw error and quit
    setup->SRJactive = 0;
    // Check if SRJ.dat exists
    std::ifstream srjfile("SRJ.dat");
    if (srjfile.is_open()) {
        // Read SRJ parameters from SRJ.dat
        // This file is arranged as: 
        // 32  -> Number of stages
        // omega_1
        // omega_2 
        // ...
        srjfile >> setup->SRJsize;
        setup->SRJomega = new real_t[setup->SRJsize];
        for (int i = 0; i < setup->SRJsize; i++) {
            srjfile >> setup->SRJomega[i];
        }
        srjfile.close();
        setup->SRJactive = 1;
        std::cout << "SRJ parameters read from SRJ.dat: SRJsize = " 
                  << setup->SRJsize << "\n";
    } else {
        setup->SRJactive = 0;
    }
    if (setup->method == METHOD_SRJ && setup->SRJactive == 0) {
        std::cerr << "ERROR: Method is SRJ but SRJ.dat not found or SRJ parameters not set!\n";
        return -1;
    }
    
    return 0;
}
