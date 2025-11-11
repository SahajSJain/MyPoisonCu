#include "structs.cuh"

Restrictor::Restrictor(int N, int Nc) : N(N), Nc(Nc), Nb(N + 2), Ncb(Nc + 2), Ntotal((Nc + 2) * (Nc + 2)) {
    // Allocate host memory (initialized to 0)
    this->R = (real_t*)calloc(this->Ntotal, sizeof(real_t));
    if (!this->R) {
        fprintf(stderr, "ERROR: Failed to allocate host memory for Restrictor (size=%d)\n", this->Ntotal);
        exit(EXIT_FAILURE);
    }
    
    // Allocate device memory based on backend
    #ifdef USE_CUDA
    allocate_cuda();
    #elif defined(USE_ACC)
    allocate_acc();
    #else
    this->R_d = nullptr; // Serial/OMP - no device allocation
    #endif
}

Restrictor::~Restrictor() {
    // Free device memory based on backend
    #ifdef USE_CUDA
    deallocate_cuda();
    #elif defined(USE_ACC)
    deallocate_acc();
    #endif
    
    // Free host memory
    if (this->R != nullptr) {
        free(this->R);
    }
}

void Restrictor::upload() {
    #ifdef USE_CUDA
    upload_cuda();
    #elif defined(USE_ACC)
    upload_acc();
    #endif
    // Do nothing for serial/OMP
}

void Restrictor::download() {
    #ifdef USE_CUDA
    download_cuda();
    #elif defined(USE_ACC)
    download_acc();
    #endif
    // Do nothing for serial/OMP
}

// OpenACC implementations
#ifdef USE_ACC
void Restrictor::allocate_acc() {
    this->R_d = nullptr; // Not used in OpenACC
    // Allocate on device and copy initial zeros using copyin
    _ACC_(acc enter data copyin(this->R[0:this->Ntotal]))
}

void Restrictor::deallocate_acc() {
    _ACC_(acc exit data delete(this->R[0:this->Ntotal]))
}

void Restrictor::upload_acc() {
    _ACC_(acc update device(this->R[0:this->Ntotal]))
}

void Restrictor::download_acc() {
    _ACC_(acc update host(this->R[0:this->Ntotal]))
}
#endif

// Stub implementations when not using CUDA
#ifndef USE_CUDA
void Restrictor::allocate_cuda() {}
void Restrictor::deallocate_cuda() {}
void Restrictor::upload_cuda() {}
void Restrictor::download_cuda() {}
#endif

// Stub implementations when not using ACC
#ifndef USE_ACC
void Restrictor::allocate_acc() {}
void Restrictor::deallocate_acc() {}
void Restrictor::upload_acc() {}
void Restrictor::download_acc() {}
#endif