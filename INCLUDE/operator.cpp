#include "structs.cuh"
#include <cstdio> // for fprintf, stderr
#include <cstdlib> // for exit

#define ONE 1.0

Operator::Operator(int N) : N(N), Nb(N + 2), Ntotal((N + 2) * (N + 2))
{
  // Allocate host memory (calloc initializes to 0)
  this->CoE = (real_t *)calloc(this->Ntotal, sizeof(real_t));
  this->CoW = (real_t *)calloc(this->Ntotal, sizeof(real_t));
  this->CoN = (real_t *)calloc(this->Ntotal, sizeof(real_t));
  this->CoS = (real_t *)calloc(this->Ntotal, sizeof(real_t));
  this->CoP = (real_t *)calloc(this->Ntotal, sizeof(real_t));
  this->CoPinv = (real_t *)calloc(this->Ntotal, sizeof(real_t));

  if (!this->CoE || !this->CoW || !this->CoN || !this->CoS || !this->CoP || !this->CoPinv)
  {
    fprintf(stderr, "ERROR: Failed to allocate host memory for Operator (size=%d)\n", this->Ntotal);
    exit(EXIT_FAILURE);
  }

  // Initialize CoP to 1.0 on host
  for (int i = 0; i < this->Ntotal; i++)
  {
    this->CoP[i] = ONE;
    this->CoPinv[i] = ONE;
  }

  // Allocate device memory based on backend
  #ifdef USE_CUDA
  allocate_cuda();
  #elif defined(USE_ACC)
  allocate_acc();
  #else
  // Serial/OMP - set device pointers to nullptr
  this->CoE_d = nullptr;
  this->CoW_d = nullptr;
  this->CoN_d = nullptr;
  this->CoS_d = nullptr;
  this->CoP_d = nullptr;
  this->CoPinv_d = nullptr;
  #endif
}

Operator::~Operator()
{
  // Free device memory based on backend
  #ifdef USE_CUDA
  deallocate_cuda();
  #elif defined(USE_ACC)
  deallocate_acc();
  #endif

  // Free host memory
  if (this->CoE != nullptr)
  {
    free(this->CoE);
  }
  if (this->CoW != nullptr)
  {
    free(this->CoW);
  }
  if (this->CoN != nullptr)
  {
    free(this->CoN);
  }
  if (this->CoS != nullptr)
  {
    free(this->CoS);
  }
  if (this->CoP != nullptr)
  {
    free(this->CoP);
  }
  if (this->CoPinv != nullptr)
  {
    free(this->CoPinv);
  }
}

void Operator::upload()
{
  #ifdef USE_CUDA
  upload_cuda();
  #elif defined(USE_ACC)
  upload_acc();
  #endif
  // Do nothing for serial/OMP
}

void Operator::download()
{
  #ifdef USE_CUDA
  download_cuda();
  #elif defined(USE_ACC)
  download_acc();
  #endif
  // Do nothing for serial/OMP
}

// OpenACC implementations
#ifdef USE_ACC

void Operator::allocate_acc()
{
  // Set device pointers to nullptr (not used in OpenACC)
  this->CoE_d = nullptr;
  this->CoW_d = nullptr;
  this->CoN_d = nullptr;
  this->CoS_d = nullptr;
  this->CoP_d = nullptr;
  this->CoPinv_d = nullptr;

  // Allocate on device and copy initial values using copyin
  _ACC_(acc enter data copyin(this->CoP [0:this->Ntotal], \
                              this->CoPinv [0:this->Ntotal], \
                              this->CoE [0:this->Ntotal], \
                              this->CoW [0:this->Ntotal], \
                              this->CoN [0:this->Ntotal], \
                              this->CoS [0:this->Ntotal]))
}


void Operator::deallocate_acc()
{
  _ACC_(acc exit data delete (this->CoP [0:this->Ntotal], \
                              this->CoPinv [0:this->Ntotal], \
                              this->CoE [0:this->Ntotal], \
                              this->CoW [0:this->Ntotal], \
                              this->CoN [0:this->Ntotal], \
                              this->CoS [0:this->Ntotal]))
}

void Operator::upload_acc()
{
  _ACC_(acc update device(this->CoP [0:this->Ntotal], \
                          this->CoPinv [0:this->Ntotal], \
                          this->CoE [0:this->Ntotal], \
                          this->CoW [0:this->Ntotal], \
                          this->CoN [0:this->Ntotal], \
                          this->CoS [0:this->Ntotal]))
}

void Operator::download_acc()
{
  _ACC_(acc update host(this->CoP [0:this->Ntotal], \
                        this->CoPinv [0:this->Ntotal], \
                        this->CoE [0:this->Ntotal], \
                        this->CoW [0:this->Ntotal], \
                        this->CoN [0:this->Ntotal], \
                        this->CoS [0:this->Ntotal]))
}
#endif

// Stub implementations when not using CUDA
#ifndef USE_CUDA
void Operator::allocate_cuda() {}
void Operator::deallocate_cuda() {}
void Operator::upload_cuda() {}
void Operator::download_cuda() {}
#endif

// Stub implementations when not using ACC
#ifndef USE_ACC
void Operator::allocate_acc() {}
void Operator::deallocate_acc() {}
void Operator::upload_acc() {}
void Operator::download_acc() {}
#endif