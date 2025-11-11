// calculators_host.cpp

#include "calculators.cuh"
#include "../INCLUDE/structs.cuh"
#include <time.h>

// Index macros for 2D grid with boundaries
#define IDX(i, j, Nb) ((i) * (Nb) + (j))
#define EAST IDX(i + 1, j, Nb)
#define WEST IDX(i - 1, j, Nb)
#define NORTH IDX(i, j + 1, Nb)
#define SOUTH IDX(i, j - 1, Nb)
#define CENTER IDX(i, j, Nb)

// Timer function - returns current time in seconds
double timer() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void calculate_residual_host(Field<real_t> &u, Field<real_t> &rhs, Operator &op, Field<real_t> &result)
{
  int i, j;
  int N = op.N;   // number of internal grid points (should match phi and result)
  int Nb = op.Nb; // number of grid points including boundaries

  // Loop over internal points only (excluding boundaries)
  _ACC_(acc data present(u.u[0 : Nb * Nb], \
                         op.CoE[0 : Nb * Nb], \
                         op.CoW[0 : Nb * Nb], \
                         op.CoN[0 : Nb * Nb], \
                         op.CoS[0 : Nb * Nb], \
                         op.CoP[0 : Nb * Nb], \
                         rhs.u[0 : Nb * Nb], \
                         result.u[0 : Nb * Nb]))
  {
  _ACC_(acc parallel loop gang vector collapse(2) private(i, j))
    _OMP_(omp parallel for collapse(2) private(i, j)) 
    for (i = 1; i <= N; i++)
    {
      for (j = 1; j <= N; j++)
      {
        // Residual calculation: result = rhs - (A * u)
        result.u[CENTER] = rhs.u[CENTER]
                            - (op.CoP[CENTER] * u.u[CENTER]
                             + op.CoE[CENTER] * u.u[EAST]
                             + op.CoW[CENTER] * u.u[WEST]
                             + op.CoN[CENTER] * u.u[NORTH]
                             + op.CoS[CENTER] * u.u[SOUTH]);
       
      }
    }
  }
}

void calculate_residual_norm_host(Field<real_t> &u, Field<real_t> &rhs, Operator &op, 
                             Field<real_t> &result, real_t &norm)
{
  int i, j;
  int N = op.N;   // number of internal grid points
  int Nb = op.Nb; // number of grid points including boundaries
  
  real_t local_norm = ZERO; 
  // use L1 norm: sum of absolute values divided by total points
  // Loop over internal points only (excluding boundaries)
  _ACC_(acc data present(u.u[0 : Nb * Nb], \
                         op.CoE[0 : Nb * Nb], \
                         op.CoW[0 : Nb * Nb], \
                         op.CoN[0 : Nb * Nb], \
                         op.CoS[0 : Nb * Nb], \
                         op.CoP[0 : Nb * Nb], \
                         rhs.u[0 : Nb * Nb], \
                         result.u[0 : Nb * Nb]) \
           copy(local_norm))
  {
  _ACC_(acc parallel loop gang vector collapse(2) private(i, j) reduction(+ : local_norm))
    _OMP_(omp parallel for collapse(2) private(i, j) reduction(+ : local_norm))
    for (i = 1; i <= N; i++)
    {
      for (j = 1; j <= N; j++)
      {
        // Residual calculation: result = rhs - (A * u)
        result.u[CENTER] = rhs.u[CENTER]
                            - (op.CoP[CENTER] * u.u[CENTER]
                             + op.CoE[CENTER] * u.u[EAST]
                             + op.CoW[CENTER] * u.u[WEST]
                             + op.CoN[CENTER] * u.u[NORTH]
                             + op.CoS[CENTER] * u.u[SOUTH]);
        real_t abs_residual = FABS(result.u[CENTER]);
        local_norm += abs_residual;
      }
    }
  }
  
  // Normalize by total number of grid points and store in result parameter
  norm = local_norm / (N * N);
}

void calculate_opinv_dot_product_host(Field<real_t> &u1, Field<real_t> &u2, Operator &op, real_t &result)
{
  int i, j;
  int N = u1.N;   // number of internal grid points
  int Nb = u1.Nb; // number of grid points including boundaries
  
  real_t dotproduct = ZERO; 
  // use L1 norm: sum of absolute values divided by total points
  // Loop over internal points only (excluding boundaries)
  _ACC_(acc data present(u1.u[0 : Nb * Nb], \
                         u2.u[0 : Nb * Nb], \
                         op.CoPinv[0 : Nb * Nb]) \
           copy(dotproduct))
  {
  _ACC_(acc parallel loop gang vector collapse(2) private(i, j) reduction(+ : dotproduct))
    _OMP_(omp parallel for collapse(2) private(i, j) reduction(+ : dotproduct))
    for (i = 1; i <= N; i++)
    {
      for (j = 1; j <= N; j++)
      {
        // we are able to do this because the grid is uniform. 
        // maybe use integral for non uniform grid? 
        // no need to bother for now 
        // this calculates : K^-1 u1 . K^-1 u2 
        dotproduct += u1.u[CENTER] * u2.u[CENTER] * op.CoPinv[CENTER] * op.CoPinv[CENTER];
      }
    }
  }

  result = dotproduct;
}

void calculate_matrix_vector_host(Field<real_t> &phi, Operator &op, Field<real_t> &result)
{
  int i, j;
  int N = op.N;   // number of internal grid points (should match phi and result)
  int Nb = op.Nb; // number of grid points including boundaries

  // Loop over internal points only (excluding boundaries)
  _ACC_(acc data present(phi.u[0 : Nb * Nb], \
                         op.CoE[0 : Nb * Nb], \
                         op.CoW[0 : Nb * Nb], \
                         op.CoN[0 : Nb * Nb], \
                         op.CoS[0 : Nb * Nb], \
                         op.CoP[0 : Nb * Nb], \
                         result.u[0 : Nb * Nb]))
  {
  _ACC_(acc parallel loop gang vector collapse(2) private(i, j))
    _OMP_(omp parallel for collapse(2) private(i, j)) 
    for (i = 1; i <= N; i++)
    {
      for (j = 1; j <= N; j++)
      {
        // 5-point stencil: result = CoP * phi_center + CoE * phi_east + ...
        result.u[CENTER] = op.CoP[CENTER] * phi.u[CENTER] +
                            op.CoE[CENTER] * phi.u[EAST] +
                            op.CoW[CENTER] * phi.u[WEST] +
                            op.CoN[CENTER] * phi.u[NORTH] +
                            op.CoS[CENTER] * phi.u[SOUTH];
      }
    }
  }
}

void calculate_dot_product_host(Field<real_t> &u1, Field<real_t> &u2, real_t &result)
{
  int i, j;
  int N = u1.N;   // number of internal grid points
  int Nb = u1.Nb; // number of grid points including boundaries
  
  real_t dotproduct = ZERO; 
  // use L1 norm: sum of absolute values divided by total points
  // Loop over internal points only (excluding boundaries)
  _ACC_(acc data present(u1.u[0 : Nb * Nb], \
                         u2.u[0 : Nb * Nb]) \
           copy(dotproduct))
  {
  _ACC_(acc parallel loop gang vector collapse(2) private(i, j) reduction(+ : dotproduct))
    _OMP_(omp parallel for collapse(2) private(i, j) reduction(+ : dotproduct))
    for (i = 1; i <= N; i++)
    {
      for (j = 1; j <= N; j++)
      {
        // we are able to do this because the grid is uniform. 
        // maybe use integral for non uniform grid? 
        // no need to bother for now 
        dotproduct += u1.u[CENTER] * u2.u[CENTER];
      }
    }
  }

  result = dotproduct;
}

void calculate_boundary_values_host(Field<real_t> &phi, Solver &solver)
{
  // This function applies boundary conditions to ghost points 
  int i, j;
  int N = solver.N;   // number of internal grid points
  int Nb = solver.Nb; // number of grid points including boundaries

  // Single data region for all boundary operations
  _ACC_(acc data present(phi.u[0 : Nb * Nb]))
  {
  // Apply boundary conditions to ghost/boundary points (skip corners)
  // East boundary
  switch (solver.bc_east)
  {
    case BC_PERIODIC: // Periodic BC
_OMP_(omp parallel for private(j))
_ACC_(acc parallel loop gang vector private(j))
      for (j = 1; j <= N; j++)
      {
        phi.u[IDX(N + 1, j, Nb)] = phi.u[IDX(1, j, Nb)];
      } // end for j (East BC_PERIODIC)
      break;
    case BC_NEUMANN: // Neumann BC
_OMP_(omp parallel for private(j))
_ACC_(acc parallel loop gang vector private(j))
      for (j = 1; j <= N; j++)
      {
        phi.u[IDX(N + 1, j, Nb)] = phi.u[IDX(N, j, Nb)]; // homogeneous Neumann BC
      } // end for j (East BC_NEUMANN)
      break;
    case BC_DIRICHLET: // Dirichlet BC
_OMP_(omp parallel for private(j))
_ACC_(acc parallel loop gang vector private(j))
      for (j = 1; j <= N; j++)
      {
        phi.u[IDX(N + 1, j, Nb)] = 2.0 * solver.valbc_east - phi.u[IDX(N, j, Nb)];
      } // end for j (East BC_DIRICHLET)
      break;
  } // end switch bc_east
  
  // West boundary
  switch (solver.bc_west)
  {
    case BC_PERIODIC: // Periodic BC
_OMP_(omp parallel for private(j))
_ACC_(acc parallel loop gang vector private(j))
      for (j = 1; j <= N; j++)
      {
        phi.u[IDX(0, j, Nb)] = phi.u[IDX(N, j, Nb)];
      } // end for j (West BC_PERIODIC)
      break;
    case BC_NEUMANN: // Neumann BC
_OMP_(omp parallel for private(j))
_ACC_(acc parallel loop gang vector private(j))
      for (j = 1; j <= N; j++)
      {
        phi.u[IDX(0, j, Nb)] = phi.u[IDX(1, j, Nb)]; // homogeneous Neumann BC
      } // end for j (West BC_NEUMANN)
      break;
    case BC_DIRICHLET: // Dirichlet BC
_OMP_(omp parallel for private(j))
_ACC_(acc parallel loop gang vector private(j))
      for (j = 1; j <= N; j++)
      {
        phi.u[IDX(0, j, Nb)] = 2.0 * solver.valbc_west - phi.u[IDX(1, j, Nb)];
      } // end for j (West BC_DIRICHLET)
      break;
  } // end switch bc_west
  
  // North boundary
  switch (solver.bc_north)
  {
    case BC_PERIODIC: // Periodic BC
_OMP_(omp parallel for private(i))
_ACC_(acc parallel loop gang vector private(i))
      for (i = 1; i <= N; i++)
      {
        phi.u[IDX(i, N + 1, Nb)] = phi.u[IDX(i, 1, Nb)];
      } // end for i (North BC_PERIODIC)
      break;
    case BC_NEUMANN: // Neumann BC
_OMP_(omp parallel for private(i))
_ACC_(acc parallel loop gang vector private(i))
      for (i = 1; i <= N; i++)
      {
        phi.u[IDX(i, N + 1, Nb)] = phi.u[IDX(i, N, Nb)]; // homogeneous Neumann BC
      } // end for i (North BC_NEUMANN)
      break;
    case BC_DIRICHLET: // Dirichlet BC
_OMP_(omp parallel for private(i))
_ACC_(acc parallel loop gang vector private(i))
      for (i = 1; i <= N; i++)
      {
        phi.u[IDX(i, N + 1, Nb)] = 2.0 * solver.valbc_north - phi.u[IDX(i, N, Nb)];
      } // end for i (North BC_DIRICHLET)
      break;
  } // end switch bc_north
  
  // South boundary
  switch (solver.bc_south)
  {
    case BC_PERIODIC: // Periodic BC
_OMP_(omp parallel for private(i))
_ACC_(acc parallel loop gang vector private(i))
      for (i = 1; i <= N; i++)
      {
        phi.u[IDX(i, 0, Nb)] = phi.u[IDX(i, N, Nb)];
      } // end for i (South BC_PERIODIC)
      break;
    case BC_NEUMANN: // Neumann BC
_OMP_(omp parallel for private(i))
_ACC_(acc parallel loop gang vector private(i))
      for (i = 1; i <= N; i++)
      {
        phi.u[IDX(i, 0, Nb)] = phi.u[IDX(i, 1, Nb)]; // homogeneous Neumann BC
      } // end for i (South BC_NEUMANN)
      break;
    case BC_DIRICHLET: // Dirichlet BC
_OMP_(omp parallel for private(i))
_ACC_(acc parallel loop gang vector private(i))
      for (i = 1; i <= N; i++)
      {
        phi.u[IDX(i, 0, Nb)] = 2.0 * solver.valbc_south - phi.u[IDX(i, 1, Nb)];
      } // end for i (South BC_DIRICHLET)
      break;
  } // end switch bc_south
  } // end acc data region
} // end calculate_boundary_values

void calculate_vector_scalar_addition_host(real_t alpha, Field<real_t> &phi, real_t beta, Field<real_t> &result)
{
  int i, j;
  int N = result.N;   // number of internal grid points (should match phi and result)
  int Nb = result.Nb; // number of grid points including boundaries
  // Loop over internal points only (excluding boundaries)
  _ACC_(acc data present(phi.u[0 : Nb * Nb], \
                         result.u[0 : Nb * Nb]))
  {
  _ACC_(acc parallel loop gang vector collapse(2) private(i, j))
    _OMP_(omp parallel for collapse(2) private(i, j)) 
    for (i = 1; i <= N; i++)
    {
      for (j = 1; j <= N; j++)
      {
        result.u[CENTER] = alpha * phi.u[CENTER] + beta;
      }
    }
  }
}

// For very specific reason, we will not use Field and Operator structs here 
// Thats because we want to be able to do Kinv * u without allocating extra Field and Operator structs
// Instead, we will pass raw pointers and sizes
void calculate_vector_product_host(real_t *u1, real_t *u2, real_t *result, int N, int Nb)
{
  int i, j;

  // Loop over internal points only (excluding boundaries)
  _ACC_(acc data present(u1[0 : Nb * Nb], \
                         u2[0 : Nb * Nb], \
                         result[0 : Nb * Nb]))
  {
  _ACC_(acc parallel loop gang vector collapse(2) private(i, j))
    _OMP_(omp parallel for collapse(2) private(i, j)) 
    for (i = 1; i <= N; i++)
    {
      for (j = 1; j <= N; j++)
      {
        result[CENTER] = u1[CENTER] * u2[CENTER];
      }
    }
  }
}

void calculate_vector_linear_combination_host(real_t alpha1, real_t alpha2, Field<real_t> &phi1, Field<real_t> &phi2, Field<real_t> &result)
{
  int i, j;
  int N = result.N;   // number of internal grid points (should match phi and result)
  int Nb = result.Nb; // number of grid points including boundaries
  // Loop over internal points only (excluding boundaries)
  _ACC_(acc data present(phi1.u[0 : Nb * Nb], \
                         phi2.u[0 : Nb * Nb], \
                         result.u[0 : Nb * Nb]))
  {
  _ACC_(acc parallel loop gang vector collapse(2) private(i, j))
    _OMP_(omp parallel for collapse(2) private(i, j)) 
    for (i = 1; i <= N; i++)
    {
      for (j = 1; j <= N; j++)
      {
        result.u[CENTER] = alpha1 * phi1.u[CENTER] + alpha2 * phi2.u[CENTER];
      }
    }
  }
}

void calculate_vector_addition_host(real_t alpha1, real_t alpha2, Field<real_t> &phi1, Field<real_t> &phi2, Field<real_t> &result)
{
  int i, j;
  int N = result.N;   // number of internal grid points (should match phi and result)
  int Nb = result.Nb; // number of grid points including boundaries
  // Loop over internal points only (excluding boundaries)
  _ACC_(acc data present(phi1.u[0 : Nb * Nb], \
                         phi2.u[0 : Nb * Nb], \
                         result.u[0 : Nb * Nb]))
  {
  _ACC_(acc parallel loop gang vector collapse(2) private(i, j))
    _OMP_(omp parallel for collapse(2) private(i, j)) 
    for (i = 1; i <= N; i++)
    {
      for (j = 1; j <= N; j++)
      {
        result.u[CENTER] = alpha1 * phi1.u[CENTER] + alpha2 * phi2.u[CENTER];
      }
    }
  }
}