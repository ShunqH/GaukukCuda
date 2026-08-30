// C++ headers
#include <cmath>            // std::sqrt(), std::abs()
#include <algorithm>        // std::max()  
#include <iostream>     // std::cout; std::endl; std::cerr
#include <cub/cub.cuh>

// Gaukuk dependence
#include "eos.cuh"
#include "../utils/read_config.hpp" // Config 

namespace Gaukuk{

// adiabatic equation of state
EquationOfState::EquationOfState(){
    // read adiabatic index gamma from input file
    gamma_ = Config::getInstance().get("gamma") ; 
    gm1Rec_ = 1.0 / (gamma_ - 1.0);
}

__device__ Real ConsToPrimCore(const CArrayView<Real>& deviConsView, 
                               CArrayView<Real>& deviPrimView, 
                               Real gm1, Real dmin, Real pmin,
                               int i, int j, int k, bool ifCalCmax){
    Real consDen = deviConsView(DEN, k, j, i); 
    Real consMtx = deviConsView(MTX, k, j, i); 
    Real consMty = deviConsView(MTY, k, j, i); 
    Real consMtz = deviConsView(MTZ, k, j, i); 
    Real consEng = deviConsView(ENG, k, j, i); 

    Real& primDen = deviPrimView(DEN, k, j, i); 
    Real& primVlx = deviPrimView(VLX, k, j, i); 
    Real& primVly = deviPrimView(VLY, k, j, i); 
    Real& primVlz = deviPrimView(VLZ, k, j, i); 
    Real& primPre = deviPrimView(PRE, k, j, i); 

    consDen = fmax(consDen , dmin); 
    primDen = consDen; 
    Real denInv = 1.0/consDen; 
    primVlx = consMtx * denInv; 
    primVly = consMty * denInv; 
    primVlz = consMtz * denInv; 
    Real engKin = 0.5 * denInv * (consMtx*consMtx + consMty*consMty + consMtz*consMtz); 
    primPre = gm1 * (consEng - engKin); 
    primPre = fmax(primPre, pmin); 

    if (ifCalCmax){
        Real gm = gm1 + 1.0;
        Real cs = sqrt(gm * primPre * denInv);
        Real local = fabs(primVlx) + cs;
        local = fmax(local, fabs(primVly) + cs);
        local = fmax(local, fabs(primVlz) + cs);
        return local; 
    }
    return 0.0; 
}

__global__ 
void ConsToPrimKernel(const CArrayView<Real> deviConsView, 
                      CArrayView<Real> deviPrimView, 
                      Real gm1, Real dmin, Real pmin,
                      int il, int ir, int jl, int jr, int kl, int kr){
    int i = il + blockIdx.x * blockDim.x + threadIdx.x;
    int j = jl + blockIdx.y * blockDim.y + threadIdx.y;
    int k = kl + blockIdx.z * blockDim.z + threadIdx.z;
    if (i>=il && i<ir && j>=jl && j<jr && k>=kl && k<kr){
        ConsToPrimCore(deviConsView, deviPrimView, gm1, dmin, pmin, i, j, k, false);
    }
}

__global__ 
void ConsToPrimCmaxKernel(const CArrayView<Real> deviConsView, 
                          CArrayView<Real> deviPrimView, 
                          Real *deviCmax, 
                          Real gm1, Real dmin, Real pmin,
                          int il, int ir, int jl, int jr, int kl, int kr){
    using BlockReduce = cub::BlockReduce<Real, BLOCK_SIZE>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    Real local = 1e-16; 
    
    int i = il + blockIdx.x * blockDim.x + threadIdx.x;
    int j = jl + blockIdx.y * blockDim.y + threadIdx.y;
    int k = kl + blockIdx.z * blockDim.z + threadIdx.z;
    if (i>=il && i<ir && j>=jl && j<jr && k>=kl && k<kr){
        local = ConsToPrimCore(deviConsView, deviPrimView, gm1, dmin, pmin, i, j, k, true);
        // atomic max
        // atomicMax(deviCmax, local);
    }
    // block reduction
    Real blockMax = BlockReduce(tempStorage).Reduce(local, cub::Max());

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        atomicMax(deviCmax, blockMax);
    }
}

void EquationOfState::ConsToPrim(const CArray<Real>& deviCons, 
                                 CArray<Real>& deviPrim, 
                                 const Grid& grid, 
                                 CValue<Real>& deviCmax, bool ifCalCmax){
    int il = grid.igb;                      // first ghost cell left side
    int ir = grid.ige;                      // last ghost cell right side + 1
    int jl = grid.jgb;                      // first ghost cell left side
    int jr = grid.jge;                      // last ghost cell right side + 1
    int kl = grid.kgb;                      // first ghost cell left side
    int kr = grid.kge;                      // last ghost cell right side + 1
    if (grid.nz==1){
        kl = grid.kb;
        kr = grid.ke;
    }
    Real gm1 = gamma_ - 1.0;
    Real dmin = DENSITY_FLOOR;
    Real pmin = PRESSURE_FLOOR;

    dim3 cudaBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 cudaGrid(
        (ir - il + cudaBlock.x - 1) / cudaBlock.x,
        (jr - jl + cudaBlock.y - 1) / cudaBlock.y,
        (kr - kl + cudaBlock.z - 1) / cudaBlock.z
    );
    if (ifCalCmax){
        ConsToPrimCmaxKernel<<<cudaGrid, cudaBlock>>>(deviCons.View(), deviPrim.View(), 
                                                    deviCmax.View(), 
                                                    gm1, dmin, pmin,
                                                    il, ir, jl, jr, kl, kr); 
    }else{
        ConsToPrimKernel<<<cudaGrid, cudaBlock>>>(deviCons.View(), deviPrim.View(), 
                                                  gm1, dmin, pmin,
                                                  il, ir, jl, jr, kl, kr); 
    }
    CUDA_CHECK(cudaGetLastError());        // kernel error capture
    CUDA_CHECK(cudaDeviceSynchronize());   // kernel synchronize
}

}