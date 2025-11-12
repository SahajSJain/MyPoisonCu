// INPUTOUTPUT/inputoutput.h
#ifndef INPUTOUTPUT_H
#define INPUTOUTPUT_H

#include "../INCLUDE/structs.cuh"

#ifdef __cplusplus
extern "C" {
#endif

// Read input parameters from specified file
// Returns 0 on success, -1 on failure
int readInput(Setup* setup, const char* filename) ;

// Write Tecplot-format output file
// Returns 0 on success, non-zero on failure
int writeTecplotOutput(Setup *setup, Solver &Sol, const char *filename);


#ifdef __cplusplus
}
#endif

#endif // INPUTOUTPUT_H
