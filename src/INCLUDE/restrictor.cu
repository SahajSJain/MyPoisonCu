#include "structs.cuh"
#include <cstdio> // for fprintf, stderr
#include <cstdlib> // for exit

#ifdef USE_CUDA
#include <cuda_runtime.h>

void Restrictor::allocate_cuda() {
    cudaError_t err = cudaMalloc((void**)&this->R_d, this->Ntotal * sizeof(real_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc failed for Restrictor: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMemset(this->R_d, 0, this->Ntotal * sizeof(real_t));
}

void Restrictor::deallocate_cuda() {
    if (this->R_d != NULL) {
        cudaFree(this->R_d);
        this->R_d = NULL;
    }
}

void Restrictor::upload_cuda() {
    cudaMemcpy(this->R_d, this->R, this->Ntotal * sizeof(real_t), cudaMemcpyHostToDevice);
}

void Restrictor::download_cuda() {
    cudaMemcpy(this->R, this->R_d, this->Ntotal * sizeof(real_t), cudaMemcpyDeviceToHost);
}
#endif