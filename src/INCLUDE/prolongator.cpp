#include "structs.cuh"

Prolongator::Prolongator(int N, int Nf) {
    this->Nf = Nf;
    this->N = N;
    this->Nfb = Nf + 2; // including boundaries
    this->Nb = N + 2; // including boundaries
    this->Ntotal = this->Nfb * this->Nfb; // total size of prolongation matrix 
    
    // Allocate host memory (initialized to 0)
    this->P = (real_t*)calloc(this->Ntotal, sizeof(real_t));
    if (!this->P) {
        fprintf(stderr, "ERROR: Failed to allocate host memory for Prolongator (size=%d)\n", this->Ntotal);
        exit(EXIT_FAILURE);
    }
    
    // Allocate device memory based on backend
    #ifdef USE_CUDA
    allocate_cuda();
    #elif defined(USE_ACC)
    allocate_acc();
    #else
    this->P_d = NULL; // Serial/OMP - no device allocation
    #endif
}

Prolongator::~Prolongator() {
    // Free device memory based on backend
    #ifdef USE_CUDA
    deallocate_cuda();
    #elif defined(USE_ACC)
    deallocate_acc();
    #endif
    
    // Free host memory
    if (this->P != NULL) {
        free(this->P);
        this->P = NULL;
    }
}

void Prolongator::upload() {
    #ifdef USE_CUDA
    upload_cuda();
    #elif defined(USE_ACC)
    upload_acc();
    #endif
    // Do nothing for serial/OMP
}

void Prolongator::download() {
    #ifdef USE_CUDA
    download_cuda();
    #elif defined(USE_ACC)
    download_acc();
    #endif
    // Do nothing for serial/OMP
}

// OpenACC implementations
#ifdef USE_ACC
void Prolongator::allocate_acc() {
    this->P_d = NULL; // Not used in OpenACC
    // Allocate on device and copy initial zeros
    _ACC_(acc enter data present_or_copyin(this->P[0:this->Ntotal]))
}

void Prolongator::deallocate_acc() {
    _ACC_(acc exit data delete(this->P[0:this->Ntotal]))
}

void Prolongator::upload_acc() {
    _ACC_(acc update device(this->P[0:this->Ntotal]))
}

void Prolongator::download_acc() {
    _ACC_(acc update host(this->P[0:this->Ntotal]))
}
#endif