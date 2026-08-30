#pragma once 

// C++ headers
#include <functional>

// Gaukuk dependence
#include "../gaukuk.hpp" 
#include "../cuda_array.cuh"

namespace Gaukuk{

struct RCPLM{
public:
    static __device__ __forceinline__
    void ReconstructX(const CArrayView<Real>& deviPrimView, 
                      Real ul[NVar], Real ur[NVar], int i, int j, int k){
        int il = i; 
        int ir = i+1; 
        #pragma unroll
        for (int n=0; n<NVar; n++){
            Real u1 = deviPrimView(n, k ,j, il-1); 
            Real u2 = deviPrimView(n, k ,j, il); 
            Real u3 = deviPrimView(n, k ,j, il+1); 
            Real dul = u2 - u1; 
            Real dur = u3 - u2; 
            Real sigma = VanLeer(dul, dur); 
            ul[n] = u2 + Real(0.5)*sigma;

            u1 = deviPrimView(n, k ,j, ir-1); 
            u2 = deviPrimView(n, k ,j, ir); 
            u3 = deviPrimView(n, k ,j, ir+1); 
            dul = u2 - u1; 
            dur = u3 - u2; 
            sigma = VanLeer(dul, dur); 
            ur[n] = u2 - Real(0.5)*sigma;
        }
    }

    static __device__ __forceinline__
    void ReconstructY(const CArrayView<Real>& deviPrimView, 
                      Real ul[NVar], Real ur[NVar], int i, int j, int k){
        int jl = j; 
        int jr = j+1; 
        #pragma unroll
        for (int n=0; n<NVar; n++){
            Real u1 = deviPrimView(n, k ,jl-1, i); 
            Real u2 = deviPrimView(n, k ,jl, i); 
            Real u3 = deviPrimView(n, k ,jl+1, i); 
            Real dul = u2 - u1; 
            Real dur = u3 - u2; 
            Real sigma = VanLeer(dul, dur); 
            ul[n] = u2 + Real(0.5)*sigma;

            u1 = deviPrimView(n, k ,jr-1, i); 
            u2 = deviPrimView(n, k ,jr, i); 
            u3 = deviPrimView(n, k ,jr+1, i); 
            dul = u2 - u1; 
            dur = u3 - u2; 
            sigma = VanLeer(dul, dur); 
            ur[n] = u2 - Real(0.5)*sigma;
        }
    }

    static __device__ __forceinline__
    void ReconstructZ(const CArrayView<Real>& deviPrimView, 
                      Real ul[NVar], Real ur[NVar], int i, int j, int k){
        int kl = k; 
        int kr = k+1; 
        #pragma unroll
        for (int n=0; n<NVar; n++){
            Real u1 = deviPrimView(n, kl-1 ,j, i); 
            Real u2 = deviPrimView(n, kl ,j, i); 
            Real u3 = deviPrimView(n, kl+1 ,j, i); 
            Real dul = u2 - u1; 
            Real dur = u3 - u2; 
            Real sigma = VanLeer(dul, dur); 
            ul[n] = u2 + Real(0.5)*sigma;

            u1 = deviPrimView(n, kr-1 ,j, i); 
            u2 = deviPrimView(n, kr ,j, i); 
            u3 = deviPrimView(n, kr+1 ,j, i); 
            dul = u2 - u1; 
            dur = u3 - u2; 
            sigma = VanLeer(dul, dur); 
            ur[n] = u2 - Real(0.5)*sigma;
        }
    }
private:
    static __device__ __forceinline__
    Real Minmod(Real a, Real b){
        Real multi = a*b; 
        Real sigma = (fabs(a) < fabs(b)) ? a : b;
        return (multi <= Real(0.0)) ? Real(0.0) : sigma;
    }

    static __device__ __forceinline__
    Real VanLeer(Real a, Real b) {
        Real multi = a*b; 
        Real sigma = Real(2.0) * multi / (a + b); 
        return (multi <= Real(0.0)) ? Real(0.0) : sigma;
        // if (multi <= Real(0.0)) {
        //     return Real(0.0);
        // }
        // return Real(2.0) * multi / (a + b);
    }

    static __device__ __forceinline__
    Real MC(Real a, Real b) {
        return Minmod(Minmod(Real(2.0)*a, Real(2.0)*b), Real(0.5)*(a+b));
    }
};

}