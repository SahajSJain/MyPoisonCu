// write for tecplots

#include <iostream>
#include <fstream>
#include <string>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include "inputoutput.h"
#include "../INCLUDE/structs.cuh"

// Index macros for 2D grid with boundaries
#define IDX(i, j, Nb) ((i) * (Nb) + (j))
#define EAST IDX(i + 1, j, Nb)
#define WEST IDX(i - 1, j, Nb)
#define NORTH IDX(i, j + 1, Nb)
#define SOUTH IDX(i, j - 1, Nb)
#define CENTER IDX(i, j, Nb)



int writeTecplotOutput(Setup *setup, Solver *Sol, const char *filename)
{ 
  // Prepend TECOUT/ to filename
  std::string fullpath = std::string("TECOUT/") + filename;
  
  std::ofstream tecout(fullpath);
  if (!tecout.is_open())
  {
    std::cerr << "Error: Cannot open output file '" << filename << "'\n";
    return -1;
  }
  tecout << "Title = \"MyPoisonousCuda\" \n\n";
  tecout << "Variables = \"X\", \"Y\", \"Phi\", \"Phi_initial\" , \"Phi_error\", \"RHS\", \"Residual\" \n\n";
  tecout << "Zone T=\"MyPoisonousCuda Zone\", I=" << Sol->N << ", J=" << Sol->N << ", F=POINT \n"; 
  // write data
  int Nb = Sol->Nb;
  for(int i = 1; i <= Sol->N; i++) {
    for(int j = 1; j <= Sol->N; j++) {
      tecout << Sol->x_c[i] << " " 
             << Sol->y_c[j] << " " 
             << Sol->phi.u[CENTER] << " "
             << Sol->phi_initial.u[CENTER] << " "
             << Sol->phi_error.u[CENTER] << " "
             << Sol->rhs.u[CENTER] << " "
             << Sol->residual.u[CENTER] << "\n";
    }
  }
  tecout.close();
  return 0;
}