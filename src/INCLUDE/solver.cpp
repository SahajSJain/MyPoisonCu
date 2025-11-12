#include "structs.cuh"
#include <cstdio> // for fprintf, stderr
#include <cstdlib> // for exit
#include <iostream> // for std::cout
Solver::Solver(int N, int Nf, int Nc, int level, 
               real_t Lxs, real_t Lxe, real_t Lys, real_t Lye, int method)
    : N(N), 
      Nb(N + 2), 
      Ntotal((N + 2) * (N + 2)),
      Nc((Nc > 1) ? Nc : 1),
      Nf((Nf > 1) ? Nf : 2*N),
      level(level),
      method(method),
      Lxs(Lxs), Lxe(Lxe), Lys(Lys), Lye(Lye),
      // Initialize all fields
      phi(N),
      phi_old(N),
      phi_initial(N),
      phi_error(N),
      rhs(N),
      residual(N),
      temp(N),
      A(N),
      color(N),
      alpha_E(N),
      alpha_W(N),
      alpha_N(N),
      alpha_S(N),
      P(nullptr),
      R(nullptr),
      bi_r0hat(nullptr),
      bi_p(nullptr),
      bi_v(nullptr),
      bi_y(nullptr),
      bi_h(nullptr),
      bi_s(nullptr),
      bi_t(nullptr),
      bi_z(nullptr)
{
    // Setup GPU parameters
    setupGPUParameters();
    
    // Calculate grid spacing
    dx = (Lxe - Lxs) / N;
    dy = (Lye - Lys) / N;
    
    // Allocate coordinates
    allocateCoordinates();
    
    // Initialize boundary conditions to Dirichlet
    bc_east = bc_west = bc_north = bc_south = 0;
    valbc_east = valbc_west = valbc_north = valbc_south = 0.0;
    
    // Initialize blanking fields
    initializeBlanking();
    
    // Allocate multigrid components if needed
    if (method == METHOD_MULTIGRID) {
        P = new Prolongator(N, this->Nf);
        R = new Restrictor(N, this->Nc);
    }
    
    // Allocate BiCGSTAB fields if needed
    if (method == METHOD_BICGSTAB) {
        allocateBiCGSTAB();
    }
    
    // Initialize BiCGSTAB scalars
    bis_rho0 = bis_alpha = bis_omega = bis_rho1 = bis_beta = 0.0;
}

Solver::~Solver() {
    // Free coordinates
    if (x) free(x);
    if (y) free(y);
    if (x_c) free(x_c);
    if (y_c) free(y_c);
    
    // Delete multigrid components
    delete P;
    delete R;
    
    // Delete BiCGSTAB fields
    deallocateBiCGSTAB();
}

void Solver::setupGPUParameters() {
    // Setup thread/block configuration for interior points
    numthreads = (N >= 16) ? 16 : N;
    numblocks = (N + numthreads - 1) / numthreads;
    numThreads2D[0] = numThreads2D[1] = numthreads;
    numBlocks2D[0] = numBlocks2D[1] = numblocks;
    
    // Setup thread/block configuration for boundary conditions
    numthreadsB = (Nb >= 16) ? 16 : Nb;
    numblocksB = (Nb + numthreadsB - 1) / numthreadsB;
    numThreads2DB[0] = numThreads2DB[1] = numthreadsB;
    numBlocks2DB[0] = numBlocks2DB[1] = numblocksB;
    std::cout << "  Num Threads    = " << numthreads << "\n";
    std::cout << "  Num Blocks     = " << numblocks << "\n";
    std::cout << "  Num Threads 2D = " << numThreads2D[0] << " x " << numThreads2D[1] << "=" << numThreads2D[0] * numThreads2D[1] << "\n"; 
    std::cout << "  Num Blocks 2D  = " << numBlocks2D[0]  << " x " << numBlocks2D[1]  << "=" << numBlocks2D[0] * numBlocks2D[1]   << "\n";
}

void Solver::allocateCoordinates() {
    // Allocate memory for coordinates (host only)
    x = (real_t*)malloc((N + 1) * sizeof(real_t));
    y = (real_t*)malloc((N + 1) * sizeof(real_t));
    x_c = (real_t*)malloc(Nb * sizeof(real_t));
    y_c = (real_t*)malloc(Nb * sizeof(real_t));
    
    if (!x || !y || !x_c || !y_c) {
        fprintf(stderr, "ERROR: Failed to allocate coordinate arrays\n");
        exit(EXIT_FAILURE);
    }
    
    // Calculate grid point coordinates
    for (int i = 0; i <= N; i++) {
        x[i] = Lxs + i * dx;
        y[i] = Lys + i * dy;
    }
    
    // Calculate cell center coordinates
    for (int i = 0; i < Nb; i++) {
        x_c[i] = Lxs - dx/2 + i * dx;
        y_c[i] = Lys - dy/2 + i * dy;
    }
}

void Solver::initializeBlanking() {
    // Fill all blanking fields with true (all points active)
    color.fill(true);
    alpha_E.fill(true);
    alpha_W.fill(true);
    alpha_N.fill(true);
    alpha_S.fill(true);
}

void Solver::allocateBiCGSTAB() {
    bi_r0hat = new Field<real_t>(N);
    bi_p = new Field<real_t>(N);
    bi_v = new Field<real_t>(N);
    bi_y = new Field<real_t>(N);
    bi_h = new Field<real_t>(N);
    bi_s = new Field<real_t>(N);
    bi_t = new Field<real_t>(N);
    bi_z = new Field<real_t>(N);
}

void Solver::deallocateBiCGSTAB() {
    delete bi_r0hat;
    delete bi_p;
    delete bi_v;
    delete bi_y;
    delete bi_h;
    delete bi_s;
    delete bi_t;
    delete bi_z;
}

void Solver::upload() {
    // Upload all fields
    phi.upload();
    phi_old.upload();
    phi_initial.upload();
    phi_error.upload();
    rhs.upload();
    residual.upload();
    temp.upload();
    
    // Upload operators
    A.upload();
    
    // Upload blanking fields
    color.upload();
    alpha_E.upload();
    alpha_W.upload();
    alpha_N.upload();
    alpha_S.upload();
    
    // Upload multigrid components if present
    if (P) P->upload();
    if (R) R->upload();
    
    // Upload BiCGSTAB fields if present
    if (method == METHOD_BICGSTAB) {
        bi_r0hat->upload();
        bi_p->upload();
        bi_v->upload();
        bi_y->upload();
        bi_h->upload();
        bi_s->upload();
        bi_t->upload();
        bi_z->upload();
    }
}

void Solver::download() {
    // Download all fields
    phi.download();
    phi_old.download();
    phi_initial.download();
    phi_error.download();
    rhs.download();
    residual.download();
    temp.download();
    
    // Download operators
    A.download();
    
    // Download blanking fields
    color.download();
    alpha_E.download();
    alpha_W.download();
    alpha_N.download();
    alpha_S.download();
    
    // Download multigrid components if present
    if (P) P->download();
    if (R) R->download();
    
    // Download BiCGSTAB fields if present
    if (method == METHOD_BICGSTAB) {
        bi_r0hat->download();
        bi_p->download();
        bi_v->download();
        bi_y->download();
        bi_h->download();
        bi_s->download();
        bi_t->download();
        bi_z->download();
    }
}