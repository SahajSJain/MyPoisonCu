// INCLUDE/structs.h
#ifndef TYPES_H
#define TYPES_H
#include <cuda_runtime.h>  
#include <stdbool.h> // Include this for bool, true, and false
#include <cfloat> // for FLT_MAX, FLT_MIN, etc.
#include <cmath>  // for math functions 
#include <cstdint> // for fixed width integer types

// Define precision based on compiler flag
// LLM generated 
#ifdef USE_DOUBLE
    typedef double real_t;
    typedef double2 real2_t;
    typedef double3 real3_t;
    typedef double4 real4_t;
    #define REAL_MAX 1.0e10
    #define REAL_MIN -1.0e10
    #define REAL_EPSILON DBL_EPSILON
#else
    typedef float real_t; 
    typedef float2 real2_t;
    typedef float3 real3_t;
    typedef float4 real4_t;
    #define REAL_MAX 1.0e5
    #define REAL_MIN -1.0e5
    #define REAL_EPSILON FLT_EPSILON
#endif

// Math functions that adapt to precision
#ifdef USE_DOUBLE
    #define SQRT(x) sqrt(x)
    #define SIN(x) sin(x)
    #define COS(x) cos(x)
    #define EXP(x) exp(x)
    #define LOG(x) log(x)
    #define POW(x,y) pow(x,y)
    #define FABS(x) fabs(x)
#else
    #define SQRT(x) sqrtf(x)
    #define SIN(x) sinf(x)
    #define COS(x) cosf(x)
    #define EXP(x) expf(x)
    #define LOG(x) logf(x)
    #define POW(x,y) powf(x,y)
    #define FABS(x) fabsf(x)
#endif

// Constants
#ifdef USE_DOUBLE
    #define PI 3.14159265358979323846
    #define ZERO 0.0
    #define ONE 1.0
    #define TWO 2.0
#else
    #define PI 3.14159265358979323846f
    #define ZERO 0.0f
    #define ONE 1.0f
    #define TWO 2.0f
#endif

// OpenMP and OpenACC pragma guards
// If compiled with -DUSE_OMP, enable OpenMP pragmas
#ifdef USE_OMP
    #define _OMP_(x) _Pragma(#x)
#else
    #define _OMP_(x)
#endif

// If compiled with -DUSE_ACC, enable OpenACC pragmas
#ifdef USE_ACC
    // Highly recommend using open acc when using cuda. 
    #define _ACC_(x) _Pragma(#x)
#else
    #define _ACC_(x)
#endif

#if defined(USE_ACC) && defined(USE_OMP)
    #error "Cannot use both OpenACC and OpenMP simultaneously"
#endif

// Constants
#define MAX_THREADS 256 // maximum number of threads per block
// define macros for INITIALIZATION_TYPE
#define INIT_ZERO 0
#define INIT_MANUFACTURED 1
#define INIT_RANDOM 2
// define macros for RHS_TYPE
#define RHS_ZERO 0
#define RHS_MANUFACTURED 1
#define RHS_CUSTOM 2
// define macros for boundary condition types
#define BC_DIRICHLET 0
#define BC_NEUMANN 1
#define BC_PERIODIC 2
// define macros for dirichlet BC specifier
#define BCD_USER_SPECIFIED 1
#define BCD_MANUFACTURED 2

// define macros for method types
#define METHOD_JACOBI 1
#define METHOD_SRJ 2
#define METHOD_BICGSTAB 3
#define METHOD_MULTIGRID 4

// Simulation type string for file naming
#ifdef defined(USE_CUDA)
    #define SIMTYPE "cuda"
#elif defined(USE_ACC)
    #define SIMTYPE "acc"
#elif defined(USE_OMP)
// moved USE_OMP check after USE_CUDA because we might use cuda and omp together
//   in that case it is still a cuda simulation. device functions will be used instead of host functions
    #define SIMTYPE "omp"
#else
    #define SIMTYPE "serial"
#endif

// define macros
// Data Structure for MyPoissonCu 
struct Setup{
    // These are read from the User from input.dat file 
    int N; // number of internal grid points in one direction 
    real_t Lx_start; // starting x-coordinate of the grid
    real_t Lx_end;   // ending x-coordinate of the grid
    real_t Ly_start; // starting y-coordinate of the grid
    real_t Ly_end;   // ending y-coordinate of the grid 
    int bc_east, bc_west, bc_north, bc_south; // boundary condition types for each side 
    int bcd_east, bcd_west, bcd_north, bcd_south; // dirichlet BC specifier. 
    // bcd_xxx = 1 (BCD_USER_SPECIFIED) means user specified value (value from valbc_xxx) 
    // bcd_xxx = 2 (BCD_MANUFACTURED) means manufactured solution Dirichlet (value from solfieldcalc)
    //             Note: For BCD_MANUFACTURED, valbc_xxx is set to 0 in read_input
    real_t valbc_east, valbc_west, valbc_north, valbc_south; // boundary condition values for each side 
    int num_levels; // number of multigrid levels 
    int method; // solution method (SRJ, MG, BiCGStab, etc.) 
    real_t omega; // relaxation parameter for iterative solvers (SRJ, Jacobi, smoothers)
    int maxiter; // maximum number of iterations for outer loop
    real_t tolerance; // convergence tolerance for residual norm
    int inneriter; // number of inner iterations (smoothing steps for multigrid)
    int INITIALIZATION_TYPE; // initialization type for potential field
    int RHS_TYPE; // type of right-hand side field  
    int ErrorCheck; // if you want to check error. Needs manufactured solution for RHS and BCs if dirichlet
    int SRJactive; // activate SRJ method 
    int SRJsize; // Number of stages in SRJ method
    real_t* SRJomega; // Array of relaxation parameters for SRJ method 
    // ErrorCheck = 0 means no error checking 
    // ErrorCheck = 1 means compute error norm against manufactured solution 
};  

// Data structures
template<typename T>
class Field { 
public:
    T* u;      
    T* u_d;    
    const int N;          
    const int Nb;         
    const int Ntotal;     

    // Constructor and Destructor
    Field(int N);
    ~Field();
    
    // Deep copy constructor
    Field(const Field& other);
    
    // Assignment operator for deep copy
    Field& operator=(const Field& other);
    
    // Device operations
    void upload();
    void download();
    
    // Deep copy operation
    void deepCopy(const Field& source);
    
    // Additional methods
    void fill(T value);           // Fill with a value (both host and device)
    void swap(Field& other);      // Swap contents with another field
    bool isOnDevice() const;      // Check if data is on device
    
private:
    // Backend-specific implementations
    void allocate_cuda();
    void allocate_acc();
    void deallocate_cuda();
    void deallocate_acc();
    void upload_cuda();
    void upload_acc();
    void download_cuda();
    void download_acc();
    void deepCopy_cuda(const Field& source);
    void deepCopy_acc(const Field& source);
    void fillDevice_cuda(T value);
    void fillDevice_acc(T value);
};


class Operator {
public:
    real_t* CoE;     // operator coefficients for East direction
    real_t* CoW;     // operator coefficients for West direction
    real_t* CoN;     // operator coefficients for North direction
    real_t* CoS;     // operator coefficients for South direction
    real_t* CoP;     // operator coefficients for Center (diagonal) direction
    real_t* CoPinv;  // inverse of center coefficient for use in smoothers
    // device copies
    real_t* CoE_d;     // operator coefficients for East direction on device
    real_t* CoW_d;     // operator coefficients for West direction on device
    real_t* CoN_d;     // operator coefficients for North direction on device
    real_t* CoS_d;     // operator coefficients for South direction on device
    real_t* CoP_d;     // operator coefficients for Center (diagonal) direction on device
    real_t* CoPinv_d;  // inverse of center coefficient for use in smoothers on device 

    const int N;       // number of grid points in one direction (excluding boundaries)
    const int Nb;      // total number of grid points including boundaries in one direction
    const int Ntotal;  // total number of grid points including boundaries in all directions

    // Constructor and Destructor
    Operator(int N);
    ~Operator();
    
    // Delete copy constructor and assignment operator
    Operator(const Operator&) = delete;
    Operator& operator=(const Operator&) = delete;
    
    // Device operations
    void upload();
    void download();
private:
    // Backend-specific implementations
    void allocate_cuda();
    void allocate_acc();
    void deallocate_cuda();
    void deallocate_acc();
    void upload_cuda();
    void upload_acc();
    void download_cuda();
    void download_acc();
};

// Prolongator and Restrictor for multigrid methods 
// these are always in reference to the grid we are at 
// Prolongator and Restrictor for multigrid methods 
// these are always in reference to the grid we are at 
// I have not implemented multigrid yet so its a bit moot to have these now
class Prolongator {
public:
    // prolongator interpolates from coarse grid to fine grid 
    // which means its written in reference to the coarse grid 
    // this also means while its stored on the coarse grid, its size is based on the fine grid
    real_t* P;      // prolongation operator matrix in CSR format
    real_t* P_d;    // prolongation operator matrix in CSR format on device
    const int Nf;         // number of fine grid points 
    const int N;          // number of coarse grid points 
    const int Nfb;        // number of fine grid points including boundaries 
    const int Nb;         // number of coarse grid points including boundaries 
    const int Ntotal;     // total number of point in the Prolongation operator matrix = Nf * Nf 

    // Constructor and Destructor
    Prolongator(int N, int Nf);
    ~Prolongator();
    
    // Delete copy constructor and assignment operator
    Prolongator(const Prolongator&) = delete;
    Prolongator& operator=(const Prolongator&) = delete;
    
    // Device operations
    void upload();
    void download();
    
private:
    // Backend-specific implementations
    void allocate_cuda();
    void allocate_acc();
    void deallocate_cuda();
    void deallocate_acc();
    void upload_cuda();
    void upload_acc();
    void download_cuda();
    void download_acc();
};

class Restrictor {
public:
    // restrictor interpolates from fine grid to coarse grid 
    // which means its written in reference to the fine grid 
    // this also means while its stored on the fine grid, its size is based on the coarse grid
    real_t* R;      // restriction operator matrix in CSR format
    real_t* R_d;    // restriction operator matrix in CSR format on device
    const int Nc;         // number of coarse grid points 
    const int N;          // number of fine grid points 
    const int Ncb;        // number of coarse grid points including boundaries  
    const int Nb;         // number of fine grid points including boundaries 
    const int Ntotal;     // total number of point in the Restriction operator matrix = Nc * Nc 

    // Constructor and Destructor
    Restrictor(int N, int Nc);
    ~Restrictor();
    
    // Delete copy constructor and assignment operator
    Restrictor(const Restrictor&) = delete;
    Restrictor& operator=(const Restrictor&) = delete;
    
    // Device operations
    void upload();
    void download();
    
private:
    // Backend-specific implementations
    void allocate_cuda();
    void allocate_acc();
    void deallocate_cuda();
    void deallocate_acc();
    void upload_cuda();
    void upload_acc();
    void download_cuda();
    void download_acc();
};

class Solver { 
public:
    // Current level details
    const int N;          // number of grid points in one direction (excluding boundaries)
    const int Nb;         // total number of grid points including boundaries in one direction
    const int Ntotal;     // total number of grid points including boundaries in all directions
    const int Nc;         // number of grid points on the coarser level in one direction (excluding boundaries) 
    const int Nf;         // number of grid points on the finer level in one direction (excluding boundaries) 
    const int level;      // current level index (0 = finest level)
    const int method;     // solution method (SRJ, MG, BiCGStab, etc.)
    
    // Fields
    Field<real_t> phi;           // potential field at this level
    Field<real_t> phi_old;       // potential field from previous iteration at this level
    Field<real_t> phi_initial;   // initial state of phi
    Field<real_t> phi_error;     // error field: phi_error = phi_exact - phi
    Field<real_t> rhs;           // right-hand side at this level 
    Field<real_t> residual;      // residual field: residual = rhs - A * phi
    Field<real_t> temp;          // temporary field for intermediate calculations
    
    // Operators
    Operator A;                  // operator at this level 
    Prolongator* P;              // prolongator to next finer level (nullptr if not multigrid)
    Restrictor* R;               // restrictor to next coarser level (nullptr if not multigrid)
    
    // Blanking fields
    Field<bool> color;           // blanking field
    Field<bool> alpha_E;         // blanking coefficient for East direction
    Field<bool> alpha_W;         // blanking coefficient for West direction
    Field<bool> alpha_N;         // blanking coefficient for North direction
    Field<bool> alpha_S;         // blanking coefficient for South direction
    
    // BiCGSTAB fields (nullptr if not using BiCGSTAB)
    Field<real_t>* bi_r0hat;     // shadow residual vector
    Field<real_t>* bi_p;         // search direction vector
    Field<real_t>* bi_v;         // A * y vector
    Field<real_t>* bi_y;         // preconditioned p vector
    Field<real_t>* bi_h;         // intermediate solution
    Field<real_t>* bi_s;         // intermediate vector 
    Field<real_t>* bi_t;         // A * z vector 
    Field<real_t>* bi_z;         // preconditioned s vector
    
    // BiCGSTAB scalars
    real_t bis_rho0;
    real_t bis_alpha;
    real_t bis_omega;
    real_t bis_rho1;
    real_t bis_beta;
    
    // GPU parameters
    int numthreads;
    int numblocks;
    int numThreads2D[2];
    int numBlocks2D[2];
    int numthreadsB;
    int numblocksB;
    int numThreads2DB[2];
    int numBlocks2DB[2];
    
    // Grid details
    const real_t Lxs, Lxe, Lys, Lye;
    real_t dx, dy;
    real_t* x;               // x-coordinates of grid points
    real_t* y;               // y-coordinates of grid points
    real_t* x_c;             // x-coordinates of cell centers
    real_t* y_c;             // y-coordinates of cell centers
    
    // Boundary conditions
    int bc_east, bc_west, bc_north, bc_south;
    real_t valbc_east, valbc_west, valbc_north, valbc_south;
    
    // Constructor and Destructor
    Solver(int N, int Nf, int Nc, int level, 
           real_t Lxs, real_t Lxe, real_t Lys, real_t Lye, int method);
    ~Solver();
    
    // Delete copy constructor and assignment operator
    Solver(const Solver&) = delete;
    Solver& operator=(const Solver&) = delete;
    
    // Device operations
    void upload();
    void download();

private:
    void setupGPUParameters();
    void allocateCoordinates();
    void initializeBlanking();
    void allocateBiCGSTAB();
    void deallocateBiCGSTAB();
};

#endif // TYPES_H
