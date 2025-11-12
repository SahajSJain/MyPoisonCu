// INCLUDE/field.cpp
#include "structs.cuh"
#include <algorithm>
#include <cstring>
#include <cstdio> // for fprintf, stderr
#include <cstdlib> // for exit


template<typename T>
Field<T>::Field(int N) : N(N), Nb(N + 2), Ntotal((N + 2) * (N + 2)) {
    // Allocate host memory
    this->u = (T*)calloc(this->Ntotal, sizeof(T));
    if (!this->u) {
        fprintf(stderr, "ERROR: Failed to allocate host memory for Field (size=%d)\n", this->Ntotal);
        exit(EXIT_FAILURE);
    }
    
    // Allocate device memory based on backend
    #ifdef USE_CUDA
    allocate_cuda();
    #elif defined(USE_ACC)
    allocate_acc();
    #else
    this->u_d = nullptr; // Serial/OMP - no device allocation
    #endif
}

// Deep copy constructor
template<typename T>
Field<T>::Field(const Field& other) : N(other.N), Nb(other.Nb), Ntotal(other.Ntotal) {
    // Allocate host memory
    this->u = (T*)malloc(this->Ntotal * sizeof(T));
    if (!this->u) {
        fprintf(stderr, "ERROR: Failed to allocate host memory for Field copy (size=%d)\n", this->Ntotal);
        exit(EXIT_FAILURE);
    }
    
    // Copy host data
    memcpy(this->u, other.u, this->Ntotal * sizeof(T));
    
    // Allocate and copy device memory based on backend
    #ifdef USE_CUDA
    allocate_cuda();
    deepCopy_cuda(other);
    #elif defined(USE_ACC)
    allocate_acc();
    deepCopy_acc(other);
    #else
    this->u_d = nullptr;
    #endif
}

// Assignment operator
template<typename T>
Field<T>& Field<T>::operator=(const Field& other) {
    if (this != &other) {
        deepCopy(other);
    }
    return *this;
}

template<typename T>
Field<T>::~Field() {
    // Free device memory based on backend
    #ifdef USE_CUDA
    deallocate_cuda();
    #elif defined(USE_ACC)
    deallocate_acc();
    #endif
    
    // Free host memory
    if (this->u != nullptr) {
        free(this->u);
    }
}

template<typename T>
void Field<T>::deepCopy(const Field& source) {
    if (this->Ntotal != source.Ntotal) {
        fprintf(stderr, "ERROR: Cannot deep copy fields of different sizes\n");
        exit(EXIT_FAILURE);
    }
    
    // Copy host data
    memcpy(this->u, source.u, this->Ntotal * sizeof(T));
    
    // Copy device data based on backend
    #ifdef USE_CUDA
    deepCopy_cuda(source);
    #elif defined(USE_ACC)
    deepCopy_acc(source);
    #endif
}


template<typename T>
void Field<T>::swap(Field& other) {
    if (this->Ntotal != other.Ntotal) {
        fprintf(stderr, "ERROR: Cannot swap fields of different sizes\n");
        exit(EXIT_FAILURE);
    }
    std::swap(this->u, other.u);
    std::swap(this->u_d, other.u_d);
}

template<typename T>
void Field<T>::upload() {
    #ifdef USE_CUDA
    upload_cuda();
    #elif defined(USE_ACC)
    upload_acc();
    #endif
}

template<typename T>
void Field<T>::download() {
    #ifdef USE_CUDA
    download_cuda();
    #elif defined(USE_ACC)
    download_acc();
    #endif
}


template<typename T>
void Field<T>::fill(T value) {
    // Fill host memory
    for (int i = 0; i < this->Ntotal; i++) {
        this->u[i] = value;
    }
}

// OpenACC implementations
#ifdef USE_ACC
template<typename T>
void Field<T>::allocate_acc() {
    this->u_d = nullptr; // Not used in OpenACC
    _ACC_(acc enter data copyin(this->u[0:this->Ntotal]))
}

template<typename T>
void Field<T>::deallocate_acc() {
    _ACC_(acc exit data delete(this->u[0:this->Ntotal]))
}

template<typename T>
void Field<T>::upload_acc() {
    _ACC_(acc update device(this->u[0:this->Ntotal]))
}

template<typename T>
void Field<T>::download_acc() {
    _ACC_(acc update host(this->u[0:this->Ntotal]))
}

template<typename T>
void Field<T>::deepCopy_acc(const Field& source) {
    // Download source data to host, then upload to this device
    _ACC_(acc update host(source.u[0:source.Ntotal]))
    _ACC_(acc update device(this->u[0:this->Ntotal]))
}
#endif

// Stub implementations when not using CUDA
#ifndef USE_CUDA
template<typename T>
void Field<T>::allocate_cuda() {}

template<typename T>
void Field<T>::deallocate_cuda() {}

template<typename T>
void Field<T>::upload_cuda() {}

template<typename T>
void Field<T>::download_cuda() {}

template<typename T>
void Field<T>::deepCopy_cuda(const Field& source) {}


template<typename T>
bool Field<T>::isOnDevice() const {
    #ifdef USE_ACC
    return true;  // Always assume data is on device for ACC
    #else
    return false;
    #endif
}
#endif


// Explicit template instantiations for Field<T> and all required methods
#ifdef USE_DOUBLE
template class Field<double>;
template Field<double>::Field(int);
template Field<double>::~Field();
template Field<double>::Field(const Field<double>&);
template Field<double>& Field<double>::operator=(const Field<double>&);
template void Field<double>::deepCopy(const Field<double>&);
template void Field<double>::fill(double);
template void Field<double>::upload();
template void Field<double>::download();
#endif

template class Field<bool>;
template Field<bool>::Field(int);
template Field<bool>::~Field();
template Field<bool>::Field(const Field<bool>&);
template Field<bool>& Field<bool>::operator=(const Field<bool>&);
template void Field<bool>::deepCopy(const Field<bool>&);
template void Field<bool>::fill(bool);
template void Field<bool>::upload();
template void Field<bool>::download();


