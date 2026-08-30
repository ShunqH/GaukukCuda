// Gaukuk dependence
#include "boundary.hpp"

namespace Gaukuk
{

__global__ 
void PeriodicKernel(CArrayView<Real> deviConsView, 
                    int il, int ir, int jl, int jr, int kl, int kr, 
                    int iDist, int jDist, int kDist){
    int i = il + blockIdx.x * blockDim.x + threadIdx.x;
    int j = jl + blockIdx.y * blockDim.y + threadIdx.y;
    int k = kl + blockIdx.z * blockDim.z + threadIdx.z;
    if (i>=il && i<ir && j>=jl && j<jr && k>=kl && k<kr){
        int iTarget = i + iDist; 
        int jTarget = j + jDist; 
        int kTarget = k + kDist; 
        for (int ivar = DEN; ivar <= ENG; ++ivar) {
            deviConsView(ivar, k, j, i) = deviConsView(ivar, kTarget, jTarget, iTarget);
        }
    }               
}

//   *** periodic boundary condition ***
//
// copy the edge cell of the activated zone 
// to the other side of the ghost cells
//
//------------------------------------------------------------
// X direction, left side 
void Boundary::PeriodicXL(CArray<Real>& deviCons, 
                          const Grid& grid, const EquationOfState& eos){
    int il = grid.igb;                      // first ghost cell left side
    int ir = grid.ib;                       // first activated cell 
    int jl = grid.jb;                       // first activated cell 
    int jr = grid.je;                       // first ghost cell right side
    int kl = grid.kb;                       // first activated cell 
    int kr = grid.ke;                       // first ghost cell right side
    int iDist = grid.nx;                    // distance between the activated and ghost cell 
    int jDist = 0; 
    int kDist = 0;

    dim3 cudaBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 cudaGrid(
        (ir - il + cudaBlock.x - 1) / cudaBlock.x,
        (jr - jl + cudaBlock.y - 1) / cudaBlock.y,
        (kr - kl + cudaBlock.z - 1) / cudaBlock.z
    );
    PeriodicKernel<<<cudaGrid, cudaBlock>>>(deviCons.View(), 
                                            il, ir, jl, jr, kl, kr, 
                                            iDist, jDist, kDist); 
    CUDA_CHECK(cudaGetLastError());        // kernel error capture
}

//------------------------------------------------------------
// X direction, right side 
void Boundary::PeriodicXR(CArray<Real>& deviCons, 
                          const Grid& grid, const EquationOfState& eos){
    int il = grid.ie;                       // first ghost cell right side
    int ir = grid.ige;                      // last ghost cell right side + 1
    int jl = grid.jb;                       // first activated cell 
    int jr = grid.je;                       // first ghost cell right side
    int kl = grid.kb;                       // first activated cell 
    int kr = grid.ke;                       // first ghost cell right side
    int iDist = -grid.nx;                   // distance between the activated and ghost cell 
    int jDist = 0; 
    int kDist = 0;

    dim3 cudaBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 cudaGrid(
        (ir - il + cudaBlock.x - 1) / cudaBlock.x,
        (jr - jl + cudaBlock.y - 1) / cudaBlock.y,
        (kr - kl + cudaBlock.z - 1) / cudaBlock.z
    );
    PeriodicKernel<<<cudaGrid, cudaBlock>>>(deviCons.View(), 
                                            il, ir, jl, jr, kl, kr, 
                                            iDist, jDist, kDist); 
    CUDA_CHECK(cudaGetLastError());        // kernel error capture
}

//------------------------------------------------------------
// Y direction, left side 
void Boundary::PeriodicYL(CArray<Real>& deviCons, 
                          const Grid& grid, const EquationOfState& eos){
    int il = grid.ib;                       // first activated cell 
    int ir = grid.ie;                       // first ghost cell right side
    int jl = grid.jgb;                      // first ghost cell left side 
    int jr = grid.jb;                       // first activated cell 
    int kl = grid.kb;                       // first activated cell 
    int kr = grid.ke;                       // first ghost cell right side
    int iDist = 0;                          // distance between the activated and ghost cell 
    int jDist = grid.ny; 
    int kDist = 0;

    dim3 cudaBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 cudaGrid(
        (ir - il + cudaBlock.x - 1) / cudaBlock.x,
        (jr - jl + cudaBlock.y - 1) / cudaBlock.y,
        (kr - kl + cudaBlock.z - 1) / cudaBlock.z
    );
    PeriodicKernel<<<cudaGrid, cudaBlock>>>(deviCons.View(), 
                                            il, ir, jl, jr, kl, kr, 
                                            iDist, jDist, kDist); 
    CUDA_CHECK(cudaGetLastError());        // kernel error capture
}

//------------------------------------------------------------
// Y direction, right side 
void Boundary::PeriodicYR(CArray<Real>& deviCons, 
                          const Grid& grid, const EquationOfState& eos){
    int il = grid.ib;                       // first activated cell 
    int ir = grid.ie;                       // first ghost cell right side
    int jl = grid.je;                       // first ghost cell right side 
    int jr = grid.jge;                      // last ghost cell right side + 1
    int kl = grid.kb;                       // first activated cell 
    int kr = grid.ke;                       // first ghost cell right side
    int iDist = 0;                          // distance between the activated and ghost cell 
    int jDist = -grid.ny; 
    int kDist = 0;

    dim3 cudaBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 cudaGrid(
        (ir - il + cudaBlock.x - 1) / cudaBlock.x,
        (jr - jl + cudaBlock.y - 1) / cudaBlock.y,
        (kr - kl + cudaBlock.z - 1) / cudaBlock.z
    );
    PeriodicKernel<<<cudaGrid, cudaBlock>>>(deviCons.View(), 
                                            il, ir, jl, jr, kl, kr, 
                                            iDist, jDist, kDist); 
    CUDA_CHECK(cudaGetLastError());        // kernel error capture
}

//------------------------------------------------------------
// Z direction, left side 
void Boundary::PeriodicZL(CArray<Real>& deviCons, 
                          const Grid& grid, const EquationOfState& eos){
    int il = grid.ib;                       // first activated cell 
    int ir = grid.ie;                       // first ghost cell right side
    int jl = grid.jb;                       // first activated cell 
    int jr = grid.je;                       // first ghost cell right side
    int kl = grid.kgb;                      // first ghost cell left side 
    int kr = grid.kb;                       // first activated cell 
    int iDist = 0;                          // distance between the activated and ghost cell 
    int jDist = 0; 
    int kDist = grid.nz;

    dim3 cudaBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 cudaGrid(
        (ir - il + cudaBlock.x - 1) / cudaBlock.x,
        (jr - jl + cudaBlock.y - 1) / cudaBlock.y,
        (kr - kl + cudaBlock.z - 1) / cudaBlock.z
    );
    PeriodicKernel<<<cudaGrid, cudaBlock>>>(deviCons.View(), 
                                            il, ir, jl, jr, kl, kr, 
                                            iDist, jDist, kDist); 
    CUDA_CHECK(cudaGetLastError());        // kernel error capture
}

//------------------------------------------------------------
// Z direction, right side 
void Boundary::PeriodicZR(CArray<Real>& deviCons, 
                          const Grid& grid, const EquationOfState& eos){
    int il = grid.ib;                       // first activated cell 
    int ir = grid.ie;                       // first ghost cell right side
    int jl = grid.jb;                       // first activated cell 
    int jr = grid.je;                       // first ghost cell right side
    int kl = grid.ke;                       // first ghost cell right side  
    int kr = grid.kge;                      // last ghost cell right side + 1
    int iDist = 0;                          // distance between the activated and ghost cell 
    int jDist = 0; 
    int kDist = -grid.nz;

    dim3 cudaBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 cudaGrid(
        (ir - il + cudaBlock.x - 1) / cudaBlock.x,
        (jr - jl + cudaBlock.y - 1) / cudaBlock.y,
        (kr - kl + cudaBlock.z - 1) / cudaBlock.z
    );
    PeriodicKernel<<<cudaGrid, cudaBlock>>>(deviCons.View(), 
                                            il, ir, jl, jr, kl, kr, 
                                            iDist, jDist, kDist); 
    CUDA_CHECK(cudaGetLastError());        // kernel error capture
}

}