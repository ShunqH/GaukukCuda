// C++ headers
#include <cmath>            // std::sqrt(), std::abs()
#include <algorithm>        // std::max()  
#include <iostream>     // std::cout; std::endl; std::cerr
#include <cub/cub.cuh>

// Gaukuk dependence
#include "eos.cuh"
#include "../utils/read_config.hpp" // Config 
#include "../utils/utils_cuda.cuh"

namespace Gaukuk{

struct CustomMaxOp {
    template <typename T>
    __device__ __forceinline__ T operator()(const T& a, const T& b) const {
        return (a > b) ? a : b;
    }
};

// adiabatic equation of state
EquationOfState::EquationOfState(){
    // read adiabatic index gamma from input file
    gamma_ = Config::getInstance().get("gamma") ; 
    gm1Rec_ = 1.0 / (gamma_ - 1.0);
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
    }
}

void EquationOfState::ConsToPrim(const CArray<Real>& deviCons, 
                                 CArray<Real>& deviPrim, 
                                 const Grid& grid){
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
    ConsToPrimKernel<<<cudaGrid, cudaBlock>>>(deviCons.View(), deviPrim.View(), 
                                              gm1, dmin, pmin,
                                              il, ir, jl, jr, kl, kr); 
    CUDA_CHECK(cudaGetLastError());        // kernel error capture
}


__global__ 
void CalCmaxKernel(const CArrayView<Real> deviConsView, 
             Real *deviCmax, 
             Real gm1, Real dmin, Real pmin,
             int il, int ir, int jl, int jr, int kl, int kr){
    // using BlockReduce = cub::BlockReduce<Real, BLOCK_SIZE>;
    using BlockReduce = cub::BlockReduce<Real, BLOCK_X, cub::BLOCK_REDUCE_WARP_REDUCTIONS, BLOCK_Y, BLOCK_Z>;
    __shared__ typename BlockReduce::TempStorage tempStorage;
    Real local = 1e-16; 
    
    int i = il + blockIdx.x * blockDim.x + threadIdx.x;
    int j = jl + blockIdx.y * blockDim.y + threadIdx.y;
    int k = kl + blockIdx.z * blockDim.z + threadIdx.z;
    if (i>=il && i<ir && j>=jl && j<jr && k>=kl && k<kr){
        Real consDen = deviConsView(DEN, k, j, i); 
        Real consMtx = deviConsView(MTX, k, j, i); 
        Real consMty = deviConsView(MTY, k, j, i); 
        Real consMtz = deviConsView(MTZ, k, j, i); 
        Real consEng = deviConsView(ENG, k, j, i); 

        // Clamp density to local variable
        consDen = fmax(consDen , dmin);
        Real denInv = 1.0 / consDen;
        Real primVlx = consMtx * denInv;
        Real primVly = consMty * denInv;
        Real primVlz = consMtz * denInv;

        // Kinetic energy
        Real engKin = 0.5 * (primVlx * consMtx + primVly * consMty + primVlz * consMtz);
        Real pres = gm1 * (consEng - engKin);
        pres = fmax(pres, pmin);
        Real cs = sqrt( (gm1+1.0) * pres * denInv );

        local = fabs(primVlx) + cs;
        local = fmax(local, fabs(primVly) + cs);
        local = fmax(local, fabs(primVlz) + cs);

        // atomic max
        // atomicMax(deviCmax, local);
    }
    // block reduction
    Real blockMax = BlockReduce(tempStorage).Reduce(local, CustomMaxOp());

    // if (threadIdx.x == 0 &&
    //     threadIdx.y == 0 &&
    //     threadIdx.z == 0) {

    //     printf("block (%d,%d,%d) max = %.15e\n",
    //         blockIdx.x, blockIdx.y, blockIdx.z, blockMax);
    // }

    if (threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0) {
        atomicMax(deviCmax, blockMax);
    }
}

void EquationOfState::CalCmax(const CArray<Real>& deviCons, const Grid& grid, 
                              CValue<Real>& deviCmax){
    int il = grid.ib;                       // first activated cell left side
    int ir = grid.ie;                       // last activated cell right side + 1
    int jl = grid.jb;                       // first activated cell left side
    int jr = grid.je;                       // last activated cell right side + 1
    int kl = grid.kb;                       // first activated cell left side
    int kr = grid.ke;                       // last activated cell right side + 1
    if (grid.nz==1){
        kl = grid.kb;
        kr = grid.ke;
    }
    Real gm1 = gamma_ - 1.0;
    Real dmin = DENSITY_FLOOR;
    Real pmin = PRESSURE_FLOOR;
    deviCmax.CopyFromHost(CMAX_FLOOR); 

    dim3 cudaBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 cudaGrid(
        (ir - il + cudaBlock.x - 1) / cudaBlock.x,
        (jr - jl + cudaBlock.y - 1) / cudaBlock.y,
        (kr - kl + cudaBlock.z - 1) / cudaBlock.z
    );
    CalCmaxKernel<<<cudaGrid, cudaBlock>>>(deviCons.View(), deviCmax.View(), 
                                           gm1, dmin, pmin,
                                           il, ir, jl, jr, kl, kr); 
    CUDA_CHECK(cudaGetLastError());        // kernel error capture
}

__global__ 
void PrimToConsKernel(const CArrayView<Real> deviPrimView,
                      CArrayView<Real> deviConsView,
                      Real gm1Rec, 
                      int il, int ir, int jl, int jr, int kl, int kr){
    int i = il + blockIdx.x * blockDim.x + threadIdx.x;
    int j = jl + blockIdx.y * blockDim.y + threadIdx.y;
    int k = kl + blockIdx.z * blockDim.z + threadIdx.z;
    if (i>=il && i<ir && j>=jl && j<jr && k>=kl && k<kr){
        Real& consDen = deviConsView(DEN, k, j, i); 
        Real& consMtx = deviConsView(MTX, k, j, i); 
        Real& consMty = deviConsView(MTY, k, j, i); 
        Real& consMtz = deviConsView(MTZ, k, j, i); 
        Real& consEng = deviConsView(ENG, k, j, i); 

        const Real& primDen = deviPrimView(DEN, k, j, i); 
        const Real& primVlx = deviPrimView(VLX, k, j, i); 
        const Real& primVly = deviPrimView(VLY, k, j, i); 
        const Real& primVlz = deviPrimView(VLZ, k, j, i); 
        const Real& primPre = deviPrimView(PRE, k, j, i); 

        consDen = primDen; 
        consMtx = primDen * primVlx; 
        consMty = primDen * primVly; 
        consMtz = primDen * primVlz; 
        consEng = primPre * gm1Rec 
                + 0.5 * primDen * ( primVlx*primVlx + primVly*primVly + primVlz*primVlz );
    }
}

void EquationOfState::PrimToCons(const CArray<Real>& deviPrim, 
                                 CArray<Real>& deviCons, 
                                 const Grid& grid){
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

    dim3 cudaBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 cudaGrid(
        (ir - il + cudaBlock.x - 1) / cudaBlock.x,
        (jr - jl + cudaBlock.y - 1) / cudaBlock.y,
        (kr - kl + cudaBlock.z - 1) / cudaBlock.z
    );
    PrimToConsKernel<<<cudaGrid, cudaBlock>>>(deviPrim.View(), deviCons.View(), 
                                              gm1Rec_, il, ir, jl, jr, kl, kr ); 
    CUDA_CHECK(cudaGetLastError());        // kernel error capture
}

}