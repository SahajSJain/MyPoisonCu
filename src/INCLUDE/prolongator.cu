#include "structs.cuh"

#ifdef USE_CUDA
#include <cuda_runtime.h>

void Prolongator::allocate_cuda() {
    cudaError_t err = cudaMalloc((void**)&this->P_d, this->Ntotal * sizeof(real_t));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc failed for Prolongator: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMemset(this->P_d, 0, this->Ntotal * sizeof(real_t));
}

void Prolongator::deallocate_cuda() {
    if (this->P_d != NULL) {
        cudaFree(this->P_d);
        this->P_d = NULL;
    }
}

void Prolongator::upload_cuda() {
    cudaMemcpy(this->P_d, this->P, this->Ntotal * sizeof(real_t), cudaMemcpyHostToDevice);
}

void Prolongator::download_cuda() {
    cudaMemcpy(this->P, this->P_d, this->Ntotal * sizeof(real_t), cudaMemcpyDeviceToHost);
}
#endif