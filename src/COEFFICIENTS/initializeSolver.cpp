#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <iostream>
#include <omp.h>
#include <cstring>
#ifdef USE_ACC
#include <openacc.h>
#endif
#include "../INCLUDE/structs.cuh"

#define IDX(i, j, Nb) ((i) * (Nb) + (j))
#define EAST IDX(i + 1, j, Nb)
#define WEST IDX(i - 1, j, Nb)
#define NORTH IDX(i, j + 1, Nb)
#define SOUTH IDX(i, j - 1, Nb)
#define CENTER IDX(i, j, Nb)

// Helper functions for manufactured solutions
inline real_t rhsfieldcalc(real_t x, real_t y)
{
  return -TWO * PI * PI * SIN(PI * x) * SIN(PI * y);
}

inline real_t solfieldcalc(real_t x, real_t y)
{
  return SIN(PI * x) * SIN(PI * y);
}

// Random number generation helper - host only
inline real_t generate_random_range(real_t min, real_t max)
{
  return min + (max - min) * ((real_t)rand() / RAND_MAX);
}

// Calculate Solver operators on device using OpenACC and OpenMP
void initializeSolver(Solver *Sol, Setup *setup)
{
  int N = Sol->N;
  int Nb = Sol->Nb;
  real_t dx = Sol->dx;
  real_t dy = Sol->dy;
  real_t idx2 = 1.0f / (dx * dx);
  real_t idy2 = 1.0f / (dy * dy);
  real_t x_local, y_local;
  int INIT_TYPE = setup->INITIALIZATION_TYPE;
  int RHS_TYPE = setup->RHS_TYPE;
  int p;
  real_t bcvalue = 0.0f;
  int i, j;

  // Copy BC types and values from setup to solver
  Sol->bc_east = setup->bc_east;
  Sol->bc_west = setup->bc_west;
  Sol->bc_north = setup->bc_north;
  Sol->bc_south = setup->bc_south;

  // Modify stuff for coarse levels
  if (Sol->level != 0)
  {
    RHS_TYPE = RHS_ZERO;
    INIT_TYPE = INIT_ZERO;
    Sol->valbc_east = 0.0f;
    Sol->valbc_west = 0.0f;
    Sol->valbc_north = 0.0f;
    Sol->valbc_south = 0.0f;
  }
  else
  {
    Sol->valbc_east = setup->valbc_east;
    Sol->valbc_west = setup->valbc_west;
    Sol->valbc_north = setup->valbc_north;
    Sol->valbc_south = setup->valbc_south;
  }

  // Upload all data to device at the beginning
  Sol->upload();

  // Setup blanking field and alpha values
  _ACC_(acc data present_or_copy(Sol->color.u [0:Nb * Nb]))
  {
    _OMP_(omp parallel for collapse(2) private(i, j, p))
    _ACC_(acc parallel loop gang vector collapse(2) private(i, j, p))
    for (i = 0; i < Nb; i++)
    {
      for (j = 0; j < Nb; j++)
      {
        p = CENTER;
        if (i == 0 || i == Nb - 1 || j == 0 || j == Nb - 1)
        {
          Sol->color.u[p] = false;
        }
        else
        {
          Sol->color.u[p] = true;
        }
      }
    }
  }

  // Set alpha values based on color field
  _ACC_(acc data present_or_copy(Sol->color.u [0:Nb * Nb], Sol->alpha_E.u [0:Nb * Nb], Sol->alpha_W.u [0:Nb * Nb], Sol->alpha_N.u [0:Nb * Nb], Sol->alpha_S.u [0:Nb * Nb]))
  {
    _OMP_(omp parallel for collapse(2) private(i, j, p))
    _ACC_(acc parallel loop gang vector collapse(2) private(i, j, p))
    for (i = 1; i <= N; i++)
    {
      for (j = 1; j <= N; j++)
      {
        p = CENTER;
        Sol->alpha_E.u[p] = Sol->color.u[EAST] == Sol->color.u[p];
        Sol->alpha_W.u[p] = Sol->color.u[WEST] == Sol->color.u[p];
        Sol->alpha_N.u[p] = Sol->color.u[NORTH] == Sol->color.u[p];
        Sol->alpha_S.u[p] = Sol->color.u[SOUTH] == Sol->color.u[p];
      }
    }
  }

  // Setup RHS
  if (RHS_TYPE == RHS_MANUFACTURED)
  {
    _ACC_(acc data present_or_copy(Sol->x_c [0:Nb], Sol->y_c [0:Nb], Sol->color.u [0:Nb * Nb], Sol->rhs.u [0:Nb * Nb]))
    {
      _OMP_(omp parallel for collapse(2) private(i, j, p, x_local, y_local))
      _ACC_(acc parallel loop gang vector collapse(2) private(i, j, p, x_local, y_local))
      for (i = 0; i < Nb; i++)
      {
        for (j = 0; j < Nb; j++)
        {
          p = CENTER;
          x_local = Sol->x_c[i];
          y_local = Sol->y_c[j];
          Sol->rhs.u[p] = rhsfieldcalc(x_local, y_local);
          Sol->rhs.u[p] = Sol->color.u[p] * Sol->rhs.u[p];
        }
      }
    }
  }
  else if (RHS_TYPE == RHS_CUSTOM)
  {
// Random RHS - do on host only
#ifdef USE_ACC
    _ACC_(acc update host(Sol->rhs.u [0:Nb * Nb]))
#endif
    {
      _OMP_(omp parallel for collapse(2) private(i, j, p))
      for (i = 0; i < Nb; i++)
      {
        for (j = 0; j < Nb; j++)
        {
          p = CENTER;
          Sol->rhs.u[p] = generate_random_range(-ONE, ONE);
          Sol->rhs.u[p] = Sol->color.u[p] * Sol->rhs.u[p];
        }
      }
    }
#ifdef USE_ACC
    _ACC_(acc update device(Sol->rhs.u [0:Nb * Nb]))
#endif
  }
  else
  {
    // RHS_ZERO
    _ACC_(acc data present_or_copy(Sol->rhs.u [0:Nb * Nb]))
    {
      _OMP_(omp parallel for collapse(2) private(i, j, p))
      _ACC_(acc parallel loop gang vector collapse(2) private(i, j, p))
      for (i = 0; i < Nb; i++)
      {
        for (j = 0; j < Nb; j++)
        {
          p = CENTER;
          Sol->rhs.u[p] = 0.0f;
        }
      }
    }
  }

  // Initialize potential field
  if (INIT_TYPE == INIT_MANUFACTURED)
  {
    _ACC_(acc data present_or_copy(Sol->x_c [0:Nb], Sol->y_c [0:Nb], Sol->color.u [0:Nb * Nb], Sol->phi.u [0:Nb * Nb]))
    {
      _OMP_(omp parallel for collapse(2) private(i, j, p, x_local, y_local))
      _ACC_(acc parallel loop gang vector collapse(2) private(i, j, p, x_local, y_local))
      for (i = 0; i < Nb; i++)
      {
        for (j = 0; j < Nb; j++)
        {
          p = CENTER;
          x_local = Sol->x_c[i];
          y_local = Sol->y_c[j];
          Sol->phi.u[p] = solfieldcalc(x_local, y_local);
          Sol->phi.u[p] = Sol->color.u[p] * Sol->phi.u[p];
        }
      }
    }
  }
  else if (INIT_TYPE == INIT_RANDOM)
  {
// Random initial guess - do on host only
#ifdef USE_ACC
    _ACC_(acc update host(Sol->phi.u [0:Nb * Nb]))
#endif
    {
      _OMP_(omp parallel for collapse(2) private(i, j, p))
      for (i = 0; i < Nb; i++)
      {
        for (j = 0; j < Nb; j++)
        {
          p = CENTER;
          Sol->phi.u[p] = generate_random_range(-ONE, ONE);
          Sol->phi.u[p] = Sol->color.u[p] * Sol->phi.u[p];
        }
      }
    }
#ifdef USE_ACC
    _ACC_(acc update device(Sol->phi.u [0:Nb * Nb]))
#endif
  }
  else
  {
    // INIT_ZERO
    _ACC_(acc data present_or_copy(Sol->phi.u [0:Nb * Nb]))
    {
      _OMP_(omp parallel for collapse(2) private(i, j, p))
      _ACC_(acc parallel loop gang vector collapse(2) private(i, j, p))
      for (i = 0; i < Nb; i++)
      {
        for (j = 0; j < Nb; j++)
        {
          p = CENTER;
          Sol->phi.u[p] = 0.0f;
        }
      }
    }
  }

  // Setup operator coefficients
  _ACC_(acc data present_or_copy(Sol->A.CoE [0:Nb * Nb], Sol->A.CoW [0:Nb * Nb], Sol->A.CoN [0:Nb * Nb], Sol->A.CoS [0:Nb * Nb], Sol->A.CoP [0:Nb * Nb]))
  {
    _OMP_(omp parallel for collapse(2) private(i, j, p))
    _ACC_(acc parallel loop gang vector collapse(2) private(i, j, p))
    for (i = 1; i <= N; i++)
    {
      for (j = 1; j <= N; j++)
      {
        p = CENTER;
        Sol->A.CoE[p] = idx2;
        Sol->A.CoW[p] = idx2;
        Sol->A.CoN[p] = idy2;
        Sol->A.CoS[p] = idy2;
        Sol->A.CoP[p] = -(2.0f * (idx2 + idy2));
      }
    }
  }

  // Apply boundary conditions
  // West BC
  i = 1;
  _ACC_(acc data present_or_copy(Sol->x_c [0:Nb], Sol->y_c [0:Nb], Sol->A.CoW [0:Nb * Nb], Sol->A.CoP [0:Nb * Nb], Sol->rhs.u [0:Nb * Nb])
            copyin(Sol->bc_west, Sol->valbc_west, Sol->level, setup->bcd_west))
  {
    _OMP_(omp parallel for private(j, p, bcvalue, x_local, y_local))
    _ACC_(acc parallel loop gang vector private(j, p, bcvalue, x_local, y_local))
    for (j = 1; j <= N; j++)
    {
      p = CENTER;
      if (Sol->bc_west == BC_DIRICHLET)
      {
        bcvalue = Sol->valbc_west;
        if (Sol->level == 0 && setup->bcd_west == BCD_MANUFACTURED)
        {
          x_local = Sol->x_c[i];
          y_local = Sol->y_c[j];
          bcvalue = solfieldcalc(x_local, y_local);
        }
        Sol->A.CoW[p] = 0.0f;
        Sol->A.CoP[p] -= idx2;
        Sol->rhs.u[p] -= 2.0 * idx2 * bcvalue;
      }
      if (Sol->bc_west == BC_NEUMANN)
      {
        Sol->A.CoW[p] = 0.0f;
        Sol->A.CoP[p] += idx2;
      }
    }
  }

  // East BC
  i = N;
  _ACC_(acc data present_or_copy(Sol->x_c [0:Nb], Sol->y_c [0:Nb], Sol->A.CoE [0:Nb * Nb], Sol->A.CoP [0:Nb * Nb], Sol->rhs.u [0:Nb * Nb])
            copyin(Sol->bc_east, Sol->valbc_east, Sol->level, setup->bcd_east))
  {
    _OMP_(omp parallel for private(j, p, bcvalue, x_local, y_local))
    _ACC_(acc parallel loop gang vector private(j, p, bcvalue, x_local, y_local))
    for (j = 1; j <= N; j++)
    {
      p = CENTER;
      if (Sol->bc_east == BC_DIRICHLET)
      {
        bcvalue = Sol->valbc_east;
        if (Sol->level == 0 && setup->bcd_east == BCD_MANUFACTURED)
        {
          x_local = Sol->x_c[i];
          y_local = Sol->y_c[j];
          bcvalue = solfieldcalc(x_local, y_local);
        }
        Sol->A.CoE[p] = 0.0f;
        Sol->A.CoP[p] -= idx2;
        Sol->rhs.u[p] -= 2.0 * idx2 * bcvalue;
      }
      if (Sol->bc_east == BC_NEUMANN)
      {
        Sol->A.CoE[p] = 0.0f;
        Sol->A.CoP[p] += idx2;
      }
    }
  }

  // South BC
  j = 1;
  _ACC_(acc data present_or_copy(Sol->x_c [0:Nb], Sol->y_c [0:Nb], Sol->A.CoS [0:Nb * Nb], Sol->A.CoP [0:Nb * Nb], Sol->rhs.u [0:Nb * Nb])
            copyin(Sol->bc_south, Sol->valbc_south, Sol->level, setup->bcd_south))
  {
    _OMP_(omp parallel for private(i, p, bcvalue, x_local, y_local))
    _ACC_(acc parallel loop gang vector private(i, p, bcvalue, x_local, y_local))
    for (i = 1; i <= N; i++)
    {
      p = CENTER;
      if (Sol->bc_south == BC_DIRICHLET)
      {
        bcvalue = Sol->valbc_south;
        if (Sol->level == 0 && setup->bcd_south == BCD_MANUFACTURED)
        {
          x_local = Sol->x_c[i];
          y_local = Sol->y_c[j];
          bcvalue = solfieldcalc(x_local, y_local);
        }
        Sol->A.CoS[p] = 0.0f;
        Sol->A.CoP[p] -= idy2;
        Sol->rhs.u[p] -= 2.0 * idy2 * bcvalue;
      }
      if (Sol->bc_south == BC_NEUMANN)
      {
        Sol->A.CoS[p] = 0.0f;
        Sol->A.CoP[p] += idy2;
      }
    }
  }

  // North BC
  j = N;
  _ACC_(acc data present_or_copy(Sol->x_c [0:Nb], Sol->y_c [0:Nb], Sol->A.CoN [0:Nb * Nb], Sol->A.CoP [0:Nb * Nb], Sol->rhs.u [0:Nb * Nb])
            copyin(Sol->bc_north, Sol->valbc_north, Sol->level, setup->bcd_north))
  {
    _OMP_(omp parallel for private(i, p, bcvalue, x_local, y_local))
    _ACC_(acc parallel loop gang vector private(i, p, bcvalue, x_local, y_local))
    for (i = 1; i <= N; i++)
    {
      p = CENTER;
      if (Sol->bc_north == BC_DIRICHLET)
      {
        bcvalue = Sol->valbc_north;
        if (Sol->level == 0 && setup->bcd_north == BCD_MANUFACTURED)
        {
          x_local = Sol->x_c[i];
          y_local = Sol->y_c[j];
          bcvalue = solfieldcalc(x_local, y_local);
        }
        Sol->A.CoN[p] = 0.0f;
        Sol->A.CoP[p] -= idy2;
        Sol->rhs.u[p] -= 2.0 * idy2 * bcvalue;
      }
      if (Sol->bc_north == BC_NEUMANN)
      {
        Sol->A.CoN[p] = 0.0f;
        Sol->A.CoP[p] += idy2;
      }
    }
  }

  // Get inverse of CoP for smoother
  _ACC_(acc data present_or_copy(Sol->A.CoP [0:Nb * Nb], Sol->A.CoPinv [0:Nb * Nb]))
  {
    _OMP_(omp parallel for collapse(2) private(i, j, p))
    _ACC_(acc parallel loop gang vector collapse(2) private(i, j, p))
    for (i = 1; i <= N; i++)
    {
      for (j = 1; j <= N; j++)
      {
        p = CENTER;
        if (Sol->A.CoP[p] != 0.0f)
        {
          Sol->A.CoPinv[p] = 1.0f / Sol->A.CoP[p];
        }
        else
        {
          Sol->A.CoPinv[p] = 1.0f;
        }
      }
    }
  }

  // Remove mean for neumann or periodic BC case
  int ii, jj;
  bool no_dirichlet = (Sol->bc_east != BC_DIRICHLET && Sol->bc_west != BC_DIRICHLET &&
                       Sol->bc_north != BC_DIRICHLET && Sol->bc_south != BC_DIRICHLET);

  if (no_dirichlet)
  {
    std::cout << " =========================================================================== \n";
    std::cout << "                 Note: All Neumann/Periodic BCs detected.\n";
    std::cout << "           Removing mean from RHS to ensure compatibility condition." << "\n";
    std::cout << " =========================================================================== \n";

    real_t rhs_mean = 0.0f;
    int count = 0;
    real_t rhs_max = -1.0e20f;
    real_t rhs_min = 1.0e20f;
    real_t rhs_L1 = 0.0f;

    _ACC_(acc data present_or_copy(Sol->rhs.u [0:Nb * Nb]) copy(rhs_mean, count, rhs_max, rhs_min, rhs_L1))
    {
      _OMP_(omp parallel for collapse(2) reduction(+:rhs_mean,count,rhs_L1) \
         reduction(max:rhs_max) reduction(min:rhs_min) private(ii, jj, p))
      _ACC_(acc parallel loop gang vector collapse(2) reduction(+ : rhs_mean, count, rhs_L1)
                reduction(max : rhs_max) reduction(min : rhs_min) private(ii, jj, p))
      for (ii = 1; ii <= N; ii++)
      {
        for (jj = 1; jj <= N; jj++)
        {
          p = IDX(ii, jj, Nb);
          rhs_mean += Sol->rhs.u[p];
          count++;
          rhs_L1 += fabs(Sol->rhs.u[p]);
          rhs_max = fmaxf(rhs_max, Sol->rhs.u[p]);
          rhs_min = fminf(rhs_min, Sol->rhs.u[p]);
        }
      }
    }

    rhs_mean /= (real_t)count;
    rhs_L1 /= (real_t)count;
    std::cout << "  RHS statistics before mean removal: \n";
    std::cout << "    Max value : " << rhs_max << "\n";
    std::cout << "    Min value : " << rhs_min << "\n";
    std::cout << "    L1 norm   : " << rhs_L1 << "\n";
    std::cout << "    Mean Value: " << rhs_mean << "\n";

    std::cout << "\n\n    Removing mean value of " << rhs_mean << " from RHS field. \n\n";

    _ACC_(acc data present_or_copy(Sol->rhs.u [0:Nb * Nb]))
    {
      _OMP_(omp parallel for collapse(2) private(ii, jj, p))
      _ACC_(acc parallel loop gang vector collapse(2) private(ii, jj, p))
      for (ii = 1; ii <= N; ii++)
      {
        for (jj = 1; jj <= N; jj++)
        {
          p = IDX(ii, jj, Nb);
          Sol->rhs.u[p] -= rhs_mean;
        }
      }
    }

    // Verify mean removal
    rhs_mean = 0.0f;
    count = 0;
    rhs_max = -1.0e20f;
    rhs_min = 1.0e20f;
    rhs_L1 = 0.0f;

    _ACC_(acc data present_or_copy(Sol->rhs.u [0:Nb * Nb]) copy(rhs_mean, count, rhs_max, rhs_min, rhs_L1))
    {
      _OMP_(omp parallel for collapse(2) reduction(+:rhs_mean,count,rhs_L1) \
         reduction(max:rhs_max) reduction(min:rhs_min) private(ii, jj, p))
      _ACC_(acc parallel loop gang vector collapse(2) reduction(+ : rhs_mean, count, rhs_L1)
                reduction(max : rhs_max) reduction(min : rhs_min) private(ii, jj, p))
      for (ii = 1; ii <= N; ii++)
      {
        for (jj = 1; jj <= N; jj++)
        {
          p = IDX(ii, jj, Nb);
          rhs_mean += Sol->rhs.u[p];
          count++;
          rhs_L1 += fabs(Sol->rhs.u[p]);
          rhs_max = fmaxf(rhs_max, Sol->rhs.u[p]);
          rhs_min = fminf(rhs_min, Sol->rhs.u[p]);
        }
      }
    }

    rhs_mean /= (real_t)count;
    rhs_L1 /= (real_t)count;
    std::cout << "  RHS statistics after mean removal: \n";
    std::cout << "    Max value : " << rhs_max << "\n";
    std::cout << "    Min value : " << rhs_min << "\n";
    std::cout << "    L1 norm   : " << rhs_L1 << "\n";
    std::cout << "    Mean value: " << rhs_mean << "\n";
  }

  // Store phi in phi_old and phi_initial
  Sol->phi_old.deepCopy(Sol->phi);
  Sol->phi_initial.deepCopy(Sol->phi);

// Download all data from device at the end
#ifdef USE_ACC
  Sol->download();
#endif

#ifdef USE_CUDA
  Sol->upload();
#endif
} // initializeSolver