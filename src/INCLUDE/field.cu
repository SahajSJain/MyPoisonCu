// INCLUDE/field.cu
#include "structs.cuh"

#ifdef USE_CUDA
#include <cuda_runtime.h>

template<typename T>
__global__ void fillKernel(T* data, T value, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

template<typename T>
void Field<T>::allocate_cuda() {
    cudaError_t err = cudaMalloc((void**)&this->u_d, this->Ntotal * sizeof(T));
    if (err != cudaSuccess) {
        fprintf(stderr, "ERROR: cudaMalloc failed for Field: %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    cudaMemset(this->u_d, 0, this->Ntotal * sizeof(T));
}

template<typename T>
void Field<T>::deallocate_cuda() {
    if (this->u_d != nullptr) {
        cudaFree(this->u_d);
        this->u_d = nullptr;
    }
}

template<typename T>
void Field<T>::upload_cuda() {
    cudaMemcpy(this->u_d, this->u, this->Ntotal * sizeof(T), cudaMemcpyHostToDevice);
}

template<typename T>
void Field<T>::download_cuda() {
    cudaMemcpy(this->u, this->u_d, this->Ntotal * sizeof(T), cudaMemcpyDeviceToHost);
}

template<typename T>
void Field<T>::deepCopy_cuda(const Field& source) {
    cudaMemcpy(this->u_d, source.u_d, this->Ntotal * sizeof(T), cudaMemcpyDeviceToDevice);
}

template<typename T>
bool Field<T>::isOnDevice() const {
    return (this->u_d != nullptr);
}

template<typename T>
void Field<T>::fillDevice_cuda(T value) {
    int blockSize = 256;
    int numBlocks = (this->Ntotal + blockSize - 1) / blockSize;
    fillKernel<<<numBlocks, blockSize>>>(this->u_d, value, this->Ntotal);
}

// Explicit instantiations
template class Field<real_t>;
template class Field<int>;
template class Field<bool>;
template class Field<float>;
template class Field<double>;
#endif