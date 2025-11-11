#include "structs.cuh"

#ifdef USE_CUDA
#include <cuda_runtime.h>

void Operator::allocate_cuda() {
    cudaMalloc((void**)&this->CoE_d, this->Ntotal * sizeof(real_t));
    cudaMalloc((void**)&this->CoW_d, this->Ntotal * sizeof(real_t));
    cudaMalloc((void**)&this->CoN_d, this->Ntotal * sizeof(real_t));
    cudaMalloc((void**)&this->CoS_d, this->Ntotal * sizeof(real_t));
    cudaMalloc((void**)&this->CoP_d, this->Ntotal * sizeof(real_t));
    cudaMalloc((void**)&this->CoPinv_d, this->Ntotal * sizeof(real_t));
    
    // Initialize device memory to zero
    cudaMemset(this->CoE_d, 0, this->Ntotal * sizeof(real_t));
    cudaMemset(this->CoW_d, 0, this->Ntotal * sizeof(real_t));
    cudaMemset(this->CoN_d, 0, this->Ntotal * sizeof(real_t));
    cudaMemset(this->CoS_d, 0, this->Ntotal * sizeof(real_t));
    cudaMemset(this->CoP_d, 0, this->Ntotal * sizeof(real_t));
    cudaMemset(this->CoPinv_d, 0, this->Ntotal * sizeof(real_t));
    
    // Initialize CoP_d and CoPinv_d to 1.0 using cudaMemcpy from host
    cudaMemcpy(this->CoP_d, this->CoP, this->Ntotal * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(this->CoPinv_d, this->CoPinv, this->Ntotal * sizeof(real_t), cudaMemcpyHostToDevice);
}

void Operator::deallocate_cuda() {
    if (this->CoE_d != NULL) {
        cudaFree(this->CoE_d);
        this->CoE_d = NULL;
    }
    if (this->CoW_d != NULL) {
        cudaFree(this->CoW_d);
        this->CoW_d = NULL;
    }
    if (this->CoN_d != NULL) {
        cudaFree(this->CoN_d);
        this->CoN_d = NULL;
    }
    if (this->CoS_d != NULL) {
        cudaFree(this->CoS_d);
        this->CoS_d = NULL;
    }
    if (this->CoP_d != NULL) {
        cudaFree(this->CoP_d);
        this->CoP_d = NULL;
    }
    if (this->CoPinv_d != NULL) {
        cudaFree(this->CoPinv_d);
        this->CoPinv_d = NULL;
    }
}

void Operator::upload_cuda() {
    cudaMemcpy(this->CoP_d, this->CoP, this->Ntotal * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(this->CoPinv_d, this->CoPinv, this->Ntotal * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(this->CoE_d, this->CoE, this->Ntotal * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(this->CoW_d, this->CoW, this->Ntotal * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(this->CoN_d, this->CoN, this->Ntotal * sizeof(real_t), cudaMemcpyHostToDevice);
    cudaMemcpy(this->CoS_d, this->CoS, this->Ntotal * sizeof(real_t), cudaMemcpyHostToDevice);
}

void Operator::download_cuda() {
    cudaMemcpy(this->CoP, this->CoP_d, this->Ntotal * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->CoPinv, this->CoPinv_d, this->Ntotal * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->CoE, this->CoE_d, this->Ntotal * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->CoW, this->CoW_d, this->Ntotal * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->CoN, this->CoN_d, this->Ntotal * sizeof(real_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(this->CoS, this->CoS_d, this->Ntotal * sizeof(real_t), cudaMemcpyDeviceToHost);
}
#endif