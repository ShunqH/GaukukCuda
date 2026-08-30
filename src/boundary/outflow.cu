// Gaukuk dependence
#include "boundary.hpp"

namespace Gaukuk
{

__global__ 
void OutflowCopyKernel(CArrayView<Real> deviConsView, 
                    int il, int ir, int jl, int jr, int kl, int kr, 
                    int iTarget, int jTarget, int kTarget){
    int i = il + blockIdx.x * blockDim.x + threadIdx.x;
    int j = jl + blockIdx.y * blockDim.y + threadIdx.y;
    int k = kl + blockIdx.z * blockDim.z + threadIdx.z;
    if (i>=il && i<ir && j>=jl && j<jr && k>=kl && k<kr){
        iTarget = (iTarget >= 0) ? iTarget : i; 
        jTarget = (jTarget >= 0) ? jTarget : j; 
        kTarget = (kTarget >= 0) ? kTarget : k; 
        for (int ivar = DEN; ivar <= ENG; ++ivar) {
            deviConsView(ivar, k, j, i) = deviConsView(ivar, kTarget, jTarget, iTarget);
        }
    }               
}

//   *** Outflow boundary condition ***
//
// copy the edge cell of the activated zone 
// to the other side of the ghost cells
//
//------------------------------------------------------------
// X direction, left side 
void Boundary::OutflowCopyXL(CArray<Real>& deviCons, 
                          const Grid& grid, const EquationOfState& eos){
    int il = grid.igb;                      // first ghost cell left side
    int ir = grid.ib;                       // first activated cell 
    int jl = grid.jb;                       // first activated cell 
    int jr = grid.je;                       // first ghost cell right side
    int kl = grid.kb;                       // first activated cell 
    int kr = grid.ke;                       // first ghost cell right side
    int iTarget = ir;                       // copy cell's id
    int jTarget = -1; 
    int kTarget = -1;

    dim3 cudaBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 cudaGrid(
        (ir - il + cudaBlock.x - 1) / cudaBlock.x,
        (jr - jl + cudaBlock.y - 1) / cudaBlock.y,
        (kr - kl + cudaBlock.z - 1) / cudaBlock.z
    );
    OutflowCopyKernel<<<cudaGrid, cudaBlock>>>(deviCons.View(), 
                                            il, ir, jl, jr, kl, kr, 
                                            iTarget, jTarget, kTarget); 
    CUDA_CHECK(cudaGetLastError());        // kernel error capture
}

//------------------------------------------------------------
// X direction, right side 
void Boundary::OutflowCopyXR(CArray<Real>& deviCons, 
                          const Grid& grid, const EquationOfState& eos){
    int il = grid.ie;                       // first ghost cell right side
    int ir = grid.ige;                      // last ghost cell right side + 1
    int jl = grid.jb;                       // first activated cell 
    int jr = grid.je;                       // first ghost cell right side
    int kl = grid.kb;                       // first activated cell 
    int kr = grid.ke;                       // first ghost cell right side
    int iTarget = il - 1;                   // copy cell's id
    int jTarget = -1; 
    int kTarget = -1;

    dim3 cudaBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 cudaGrid(
        (ir - il + cudaBlock.x - 1) / cudaBlock.x,
        (jr - jl + cudaBlock.y - 1) / cudaBlock.y,
        (kr - kl + cudaBlock.z - 1) / cudaBlock.z
    );
    OutflowCopyKernel<<<cudaGrid, cudaBlock>>>(deviCons.View(), 
                                            il, ir, jl, jr, kl, kr, 
                                            iTarget, jTarget, kTarget); 
    CUDA_CHECK(cudaGetLastError());        // kernel error capture
}

//------------------------------------------------------------
// Y direction, left side 
void Boundary::OutflowCopyYL(CArray<Real>& deviCons, 
                          const Grid& grid, const EquationOfState& eos){
    int il = grid.ib;                       // first activated cell 
    int ir = grid.ie;                       // first ghost cell right side
    int jl = grid.jgb;                      // first ghost cell left side 
    int jr = grid.jb;                       // first activated cell 
    int kl = grid.kb;                       // first activated cell 
    int kr = grid.ke;                       // first ghost cell right side
    int iTarget = -1;  
    int jTarget = jr;                       // copy cell's id
    int kTarget = -1;

    dim3 cudaBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 cudaGrid(
        (ir - il + cudaBlock.x - 1) / cudaBlock.x,
        (jr - jl + cudaBlock.y - 1) / cudaBlock.y,
        (kr - kl + cudaBlock.z - 1) / cudaBlock.z
    );
    OutflowCopyKernel<<<cudaGrid, cudaBlock>>>(deviCons.View(), 
                                            il, ir, jl, jr, kl, kr, 
                                            iTarget, jTarget, kTarget); 
    CUDA_CHECK(cudaGetLastError());        // kernel error capture
}

//------------------------------------------------------------
// Y direction, right side 
void Boundary::OutflowCopyYR(CArray<Real>& deviCons, 
                          const Grid& grid, const EquationOfState& eos){
    int il = grid.ib;                       // first activated cell 
    int ir = grid.ie;                       // first ghost cell right side
    int jl = grid.je;                       // first ghost cell right side 
    int jr = grid.jge;                      // last ghost cell right side + 1
    int kl = grid.kb;                       // first activated cell 
    int kr = grid.ke;                       // first ghost cell right side
    int iTarget = -1;  
    int jTarget = jl - 1;                   // copy cell's id
    int kTarget = -1;

    dim3 cudaBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 cudaGrid(
        (ir - il + cudaBlock.x - 1) / cudaBlock.x,
        (jr - jl + cudaBlock.y - 1) / cudaBlock.y,
        (kr - kl + cudaBlock.z - 1) / cudaBlock.z
    );
    OutflowCopyKernel<<<cudaGrid, cudaBlock>>>(deviCons.View(), 
                                            il, ir, jl, jr, kl, kr, 
                                            iTarget, jTarget, kTarget); 
    CUDA_CHECK(cudaGetLastError());        // kernel error capture
}

//------------------------------------------------------------
// Z direction, left side 
void Boundary::OutflowCopyZL(CArray<Real>& deviCons, 
                          const Grid& grid, const EquationOfState& eos){
    int il = grid.ib;                       // first activated cell 
    int ir = grid.ie;                       // first ghost cell right side
    int jl = grid.jb;                       // first activated cell 
    int jr = grid.je;                       // first ghost cell right side
    int kl = grid.kgb;                      // first ghost cell left side 
    int kr = grid.kb;                       // first activated cell 
    int iTarget = -1;  
    int jTarget = -1; 
    int kTarget = kr;                       // copy cell's id

    dim3 cudaBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 cudaGrid(
        (ir - il + cudaBlock.x - 1) / cudaBlock.x,
        (jr - jl + cudaBlock.y - 1) / cudaBlock.y,
        (kr - kl + cudaBlock.z - 1) / cudaBlock.z
    );
    OutflowCopyKernel<<<cudaGrid, cudaBlock>>>(deviCons.View(), 
                                            il, ir, jl, jr, kl, kr, 
                                            iTarget, jTarget, kTarget); 
    CUDA_CHECK(cudaGetLastError());        // kernel error capture
}

//------------------------------------------------------------
// Z direction, right side 
void Boundary::OutflowCopyZR(CArray<Real>& deviCons, 
                          const Grid& grid, const EquationOfState& eos){
    int il = grid.ib;                       // first activated cell 
    int ir = grid.ie;                       // first ghost cell right side
    int jl = grid.jb;                       // first activated cell 
    int jr = grid.je;                       // first ghost cell right side
    int kl = grid.ke;                       // first ghost cell right side  
    int kr = grid.kge;                      // last ghost cell right side + 1
    int iTarget = -1;  
    int jTarget = -1; 
    int kTarget = kl - 1;                   // copy cell's id

    dim3 cudaBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 cudaGrid(
        (ir - il + cudaBlock.x - 1) / cudaBlock.x,
        (jr - jl + cudaBlock.y - 1) / cudaBlock.y,
        (kr - kl + cudaBlock.z - 1) / cudaBlock.z
    );
    OutflowCopyKernel<<<cudaGrid, cudaBlock>>>(deviCons.View(), 
                                            il, ir, jl, jr, kl, kr, 
                                            iTarget, jTarget, kTarget); 
    CUDA_CHECK(cudaGetLastError());        // kernel error capture
}

}