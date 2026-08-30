#pragma once 

// C++ headers
#include <functional>

// Gaukuk dependence
#include "../gaukuk.hpp" 
#include "../cuda_array.cuh"

namespace Gaukuk{

struct RCFirstOrder{
public:
    static __device__ __forceinline__
    void ReconstructX(const CArrayView<Real>& deviPrimView, 
                      Real ul[NVar], Real ur[NVar], int i, int j, int k){
        int il = i; 
        int ir = i+1; 
        #pragma unroll
        for (int n=0; n<NVar; n++){
            ul[n] = deviPrimView(n, k ,j, il); 
            ur[n] = deviPrimView(n, k ,j, ir); 
        }
    }

    static __device__ __forceinline__
    void ReconstructY(const CArrayView<Real>& deviPrimView, 
                      Real ul[NVar], Real ur[NVar], int i, int j, int k){
        int jl = j; 
        int jr = j+1; 
        #pragma unroll
        for (int n=0; n<NVar; n++){
            ul[n] = deviPrimView(n, k ,jl, i); 
            ur[n] = deviPrimView(n, k ,jr, i); 
        }
    }

    static __device__ __forceinline__
    void ReconstructZ(const CArrayView<Real>& deviPrimView, 
                      Real ul[NVar], Real ur[NVar], int i, int j, int k){
        int kl = k; 
        int kr = k+1; 
        #pragma unroll
        for (int n=0; n<NVar; n++){
            ul[n] = deviPrimView(n, kl ,j, i); 
            ur[n] = deviPrimView(n, kr ,j, i); 
        }
    }
};

}