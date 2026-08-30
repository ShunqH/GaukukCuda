#pragma once 

#include <cuda_runtime.h> // cudaMalloc, cudaFree 
#include <cstdio>    // for error printing

namespace Gaukuk{

#define CUDA_CHECK(call) do {                          \
    cudaError_t err = call;                            \
    if (err != cudaSuccess) {                          \
        fprintf(stderr, "CUDA error at %s:%d: %s\n",   \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE);                            \
    }                                                  \
} while(0)

#ifdef __CUDACC__
__device__ __forceinline__ 
float atomicMax(float* address, float val){
    int* address_as_int = (int*)address; 
    int old = *address_as_int; 
    int assumed; 
    do {
        assumed = old; 
        if (val <= __int_as_float(assumed)){
            break; 
        }
        old = atomicCAS(address_as_int, assumed, __float_as_int(val)); 
    }while (assumed != old); 
    return __int_as_float(old);
}

__device__ __forceinline__
double atomicMax(double* address, double val){
    unsigned long long int* address_as_ull = (unsigned long long int*) address; 
    unsigned long long int old = *address_as_ull; 
    unsigned long long int assumed; 
    do{
        assumed = old; 
        if (val <= __longlong_as_double(assumed)){
            break; 
        }
        old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val)); 
    }while (assumed != old); 
    return __longlong_as_double(old); 
}
#endif

}