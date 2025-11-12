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
  int Ntotal = Sol->Ntotal;
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
#ifdef USE_ACC
#pragma acc wait
#endif

  // Setup blanking field and alpha values
  auto color_ptr = Sol->color.u;
  int Ntotal_local = Ntotal;

  _ACC_(acc data present(color_ptr [0:Ntotal_local]))
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
          color_ptr[p] = false;
        }
        else
        {
          color_ptr[p] = true;
        }
      }
    }
  }

  // Set alpha values based on color field
  auto alpha_E_ptr = Sol->alpha_E.u;
  auto alpha_W_ptr = Sol->alpha_W.u;
  auto alpha_N_ptr = Sol->alpha_N.u;
  auto alpha_S_ptr = Sol->alpha_S.u;

  _ACC_(acc data present(color_ptr [0:Ntotal], alpha_E_ptr [0:Ntotal], alpha_W_ptr [0:Ntotal], alpha_N_ptr [0:Ntotal], alpha_S_ptr [0:Ntotal]))
  {
    _OMP_(omp parallel for collapse(2) private(i, j, p))
    _ACC_(acc parallel loop gang vector collapse(2) private(i, j, p))
    for (i = 1; i <= N; i++)
    {
      for (j = 1; j <= N; j++)
      {
        p = CENTER;
        alpha_E_ptr[p] = color_ptr[EAST] == color_ptr[p];
        alpha_W_ptr[p] = color_ptr[WEST] == color_ptr[p];
        alpha_N_ptr[p] = color_ptr[NORTH] == color_ptr[p];
        alpha_S_ptr[p] = color_ptr[SOUTH] == color_ptr[p];
      }
    }
  }

  // Setup RHS
  auto rhs_ptr = Sol->rhs.u;
  auto x_c_ptr = Sol->x_c;
  auto y_c_ptr = Sol->y_c;

  if (RHS_TYPE == RHS_MANUFACTURED)
  {
    _ACC_(acc data present(x_c_ptr [0:Nb], y_c_ptr [0:Nb], color_ptr [0:Ntotal], rhs_ptr [0:Ntotal]))
    {
      _OMP_(omp parallel for collapse(2) private(i, j, p, x_local, y_local))
      _ACC_(acc parallel loop gang vector collapse(2) private(i, j, p, x_local, y_local))
      for (i = 0; i < Nb; i++)
      {
        for (j = 0; j < Nb; j++)
        {
          p = CENTER;
          x_local = x_c_ptr[i];
          y_local = y_c_ptr[j];
          rhs_ptr[p] = rhsfieldcalc(x_local, y_local);
          rhs_ptr[p] = color_ptr[p] * rhs_ptr[p];
        }
      }
    }
  }
  else if (RHS_TYPE == RHS_CUSTOM)
  {
// Random RHS - do on host only
#ifdef USE_ACC
    _ACC_(acc update host(rhs_ptr [0:Ntotal], color_ptr [0:Ntotal]))
#endif
    {
      _OMP_(omp parallel for collapse(2) private(i, j, p))
      for (i = 0; i < Nb; i++)
      {
        for (j = 0; j < Nb; j++)
        {
          p = CENTER;
          rhs_ptr[p] = generate_random_range(-ONE, ONE);
          rhs_ptr[p] = color_ptr[p] * rhs_ptr[p];
        }
      }
    }
#ifdef USE_ACC
    _ACC_(acc update device(rhs_ptr [0:Ntotal]))
#endif
  }
  else
  {
    // RHS_ZERO
    _ACC_(acc data present(rhs_ptr [0:Ntotal]))
    {
      _OMP_(omp parallel for collapse(2) private(i, j, p))
      _ACC_(acc parallel loop gang vector collapse(2) private(i, j, p))
      for (i = 0; i < Nb; i++)
      {
        for (j = 0; j < Nb; j++)
        {
          p = CENTER;
          rhs_ptr[p] = 0.0f;
        }
      }
    }
  }

  // Initialize potential field
  auto phi_ptr = Sol->phi.u;

  if (INIT_TYPE == INIT_MANUFACTURED)
  {
    _ACC_(acc data present(x_c_ptr [0:Nb], y_c_ptr [0:Nb], color_ptr [0:Ntotal], phi_ptr [0:Ntotal]))
    {
      _OMP_(omp parallel for collapse(2) private(i, j, p, x_local, y_local))
      _ACC_(acc parallel loop gang vector collapse(2) private(i, j, p, x_local, y_local))
      for (i = 0; i < Nb; i++)
      {
        for (j = 0; j < Nb; j++)
        {
          p = CENTER;
          x_local = x_c_ptr[i];
          y_local = y_c_ptr[j];
          phi_ptr[p] = solfieldcalc(x_local, y_local);
          phi_ptr[p] = color_ptr[p] * phi_ptr[p];
        }
      }
    }
  }
  else if (INIT_TYPE == INIT_RANDOM)
  {
// Random initial guess - do on host only
#ifdef USE_ACC
    _ACC_(acc update host(phi_ptr [0:Ntotal], color_ptr [0:Ntotal]))
#endif
    {
      _OMP_(omp parallel for collapse(2) private(i, j, p))
      for (i = 0; i < Nb; i++)
      {
        for (j = 0; j < Nb; j++)
        {
          p = CENTER;
          phi_ptr[p] = generate_random_range(-ONE, ONE);
          phi_ptr[p] = color_ptr[p] * phi_ptr[p];
        }
      }
    }
#ifdef USE_ACC
    _ACC_(acc update device(phi_ptr [0:Ntotal]))
#endif
  }
  else
  {
    // INIT_ZERO
    _ACC_(acc data present(phi_ptr [0:Ntotal]))
    {
      _OMP_(omp parallel for collapse(2) private(i, j, p))
      _ACC_(acc parallel loop gang vector collapse(2) private(i, j, p))
      for (i = 0; i < Nb; i++)
      {
        for (j = 0; j < Nb; j++)
        {
          p = CENTER;
          phi_ptr[p] = 0.0f;
        }
      }
    }
  }

  // Setup operator coefficients
  auto CoE_ptr = Sol->A.CoE;
  auto CoW_ptr = Sol->A.CoW;
  auto CoN_ptr = Sol->A.CoN;
  auto CoS_ptr = Sol->A.CoS;
  auto CoP_ptr = Sol->A.CoP;

  _ACC_(acc data present(CoE_ptr [0:Ntotal], CoW_ptr [0:Ntotal], CoN_ptr [0:Ntotal], CoS_ptr [0:Ntotal], CoP_ptr [0:Ntotal]))
  {
    _OMP_(omp parallel for collapse(2) private(i, j, p))
    _ACC_(acc parallel loop gang vector collapse(2) private(i, j, p))
    for (i = 1; i <= N; i++)
    {
      for (j = 1; j <= N; j++)
      {
        p = CENTER;
        CoE_ptr[p] = idx2;
        CoW_ptr[p] = idx2;
        CoN_ptr[p] = idy2;
        CoS_ptr[p] = idy2;
        CoP_ptr[p] = -(2.0f * (idx2 + idy2));
      }
    }
  }

  // Apply boundary conditions
  // Apply boundary conditions
  // Create local copies of all BC values and types
  int bc_west_local = Sol->bc_west;
  int bc_east_local = Sol->bc_east;
  int bc_north_local = Sol->bc_north;
  int bc_south_local = Sol->bc_south;
  real_t valbc_west_local = Sol->valbc_west;
  real_t valbc_east_local = Sol->valbc_east;
  real_t valbc_north_local = Sol->valbc_north;
  real_t valbc_south_local = Sol->valbc_south;
  int level_local = Sol->level;
  int bcd_west_local = setup->bcd_west;
  int bcd_east_local = setup->bcd_east;
  int bcd_north_local = setup->bcd_north;
  int bcd_south_local = setup->bcd_south;

  // West BC
  i = 1;
  _ACC_(acc data present_or_copyin(x_c_ptr [0:Nb], y_c_ptr [0:Nb])
            present(CoW_ptr [0:Ntotal], CoP_ptr [0:Ntotal], rhs_ptr [0:Ntotal]))
  {
    _OMP_(omp parallel for private(j, p, bcvalue, x_local, y_local))
    _ACC_(acc parallel loop gang vector private(j, p, bcvalue, x_local, y_local))
    for (j = 1; j <= N; j++)
    {
      p = CENTER;
      if (bc_west_local == BC_DIRICHLET)
      {
        bcvalue = valbc_west_local;
        if (level_local == 0 && bcd_west_local == BCD_MANUFACTURED)
        {
          x_local = x_c_ptr[i];
          y_local = y_c_ptr[j];
          bcvalue = solfieldcalc(x_local, y_local);
        }
        CoW_ptr[p] = 0.0f;
        CoP_ptr[p] -= idx2;
        rhs_ptr[p] -= 2.0 * idx2 * bcvalue;
      }
      if (bc_west_local == BC_NEUMANN)
      {
        CoW_ptr[p] = 0.0f;
        CoP_ptr[p] += idx2;
      }
    }
  }

  // East BC
  i = N;
  _ACC_(acc data present_or_copyin(x_c_ptr [0:Nb], y_c_ptr [0:Nb])
            present(CoE_ptr [0:Ntotal], CoP_ptr [0:Ntotal], rhs_ptr [0:Ntotal]))
  {
    _OMP_(omp parallel for private(j, p, bcvalue, x_local, y_local))
    _ACC_(acc parallel loop gang vector private(j, p, bcvalue, x_local, y_local))
    for (j = 1; j <= N; j++)
    {
      p = CENTER;
      if (bc_east_local == BC_DIRICHLET)
      {
        bcvalue = valbc_east_local;
        if (level_local == 0 && bcd_east_local == BCD_MANUFACTURED)
        {
          x_local = x_c_ptr[i];
          y_local = y_c_ptr[j];
          bcvalue = solfieldcalc(x_local, y_local);
        }
        CoE_ptr[p] = 0.0f;
        CoP_ptr[p] -= idx2;
        rhs_ptr[p] -= 2.0 * idx2 * bcvalue;
      }
      if (bc_east_local == BC_NEUMANN)
      {
        CoE_ptr[p] = 0.0f;
        CoP_ptr[p] += idx2;
      }
    }
  }

  // South BC
  j = 1;
  _ACC_(acc data present_or_copyin(x_c_ptr [0:Nb], y_c_ptr [0:Nb])
            present(CoS_ptr [0:Ntotal], CoP_ptr [0:Ntotal], rhs_ptr [0:Ntotal]))
  {
    _OMP_(omp parallel for private(i, p, bcvalue, x_local, y_local))
    _ACC_(acc parallel loop gang vector private(i, p, bcvalue, x_local, y_local))
    for (i = 1; i <= N; i++)
    {
      p = CENTER;
      if (bc_south_local == BC_DIRICHLET)
      {
        bcvalue = valbc_south_local;
        if (level_local == 0 && bcd_south_local == BCD_MANUFACTURED)
        {
          x_local = x_c_ptr[i];
          y_local = y_c_ptr[j];
          bcvalue = solfieldcalc(x_local, y_local);
        }
        CoS_ptr[p] = 0.0f;
        CoP_ptr[p] -= idy2;
        rhs_ptr[p] -= 2.0 * idy2 * bcvalue;
      }
      if (bc_south_local == BC_NEUMANN)
      {
        CoS_ptr[p] = 0.0f;
        CoP_ptr[p] += idy2;
      }
    }
  }

  // North BC
  j = N;
  _ACC_(acc data present_or_copyin(x_c_ptr [0:Nb], y_c_ptr [0:Nb])
            present(CoN_ptr [0:Ntotal], CoP_ptr [0:Ntotal], rhs_ptr [0:Ntotal]))
  {
    _OMP_(omp parallel for private(i, p, bcvalue, x_local, y_local))
    _ACC_(acc parallel loop gang vector private(i, p, bcvalue, x_local, y_local))
    for (i = 1; i <= N; i++)
    {
      p = CENTER;
      if (bc_north_local == BC_DIRICHLET)
      {
        bcvalue = valbc_north_local;
        if (level_local == 0 && bcd_north_local == BCD_MANUFACTURED)
        {
          x_local = x_c_ptr[i];
          y_local = y_c_ptr[j];
          bcvalue = solfieldcalc(x_local, y_local);
        }
        CoN_ptr[p] = 0.0f;
        CoP_ptr[p] -= idy2;
        rhs_ptr[p] -= 2.0 * idy2 * bcvalue;
      }
      if (bc_north_local == BC_NEUMANN)
      {
        CoN_ptr[p] = 0.0f;
        CoP_ptr[p] += idy2;
      }
    }
  }

  // Get inverse of CoP for smoother
  auto CoPinv_ptr = Sol->A.CoPinv;

  _ACC_(acc data present(CoP_ptr [0:Ntotal], CoPinv_ptr [0:Ntotal]))
  {
    _OMP_(omp parallel for collapse(2) private(i, j, p))
    _ACC_(acc parallel loop gang vector collapse(2) private(i, j, p))
    for (i = 1; i <= N; i++)
    {
      for (j = 1; j <= N; j++)
      {
        p = CENTER;
        if (CoP_ptr[p] != 0.0f)
        {
          CoPinv_ptr[p] = 1.0f / CoP_ptr[p];
        }
        else
        {
          CoPinv_ptr[p] = 1.0f;
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

    _ACC_(acc data present(rhs_ptr [0:Ntotal]) copy(rhs_mean, count, rhs_max, rhs_min, rhs_L1))
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
          rhs_mean += rhs_ptr[p];
          count++;
          rhs_L1 += fabs(rhs_ptr[p]);
          rhs_max = fmaxf(rhs_max, rhs_ptr[p]);
          rhs_min = fminf(rhs_min, rhs_ptr[p]);
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

    _ACC_(acc data present(rhs_ptr [0:Ntotal]))
    {
      _OMP_(omp parallel for collapse(2) private(ii, jj, p))
      _ACC_(acc parallel loop gang vector collapse(2) private(ii, jj, p))
      for (ii = 1; ii <= N; ii++)
      {
        for (jj = 1; jj <= N; jj++)
        {
          p = IDX(ii, jj, Nb);
          rhs_ptr[p] -= rhs_mean;
        }
      }
    }

    // Verify mean removal
    rhs_mean = 0.0f;
    count = 0;
    rhs_max = -1.0e20f;
    rhs_min = 1.0e20f;
    rhs_L1 = 0.0f;

    _ACC_(acc data present(rhs_ptr [0:Ntotal]) copy(rhs_mean, count, rhs_max, rhs_min, rhs_L1))
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
          rhs_mean += rhs_ptr[p];
          count++;
          rhs_L1 += fabs(rhs_ptr[p]);
          rhs_max = fmaxf(rhs_max, rhs_ptr[p]);
          rhs_min = fminf(rhs_min, rhs_ptr[p]);
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