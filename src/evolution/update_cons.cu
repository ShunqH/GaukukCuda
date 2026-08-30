// Gaukuk dependence
#include "../cuda_array.cuh"
#include "../sim.hpp" 
#include "../reconstruction/reconstruction.hpp"
#include "../flux/flux.hpp"

namespace{

enum Dimension {TwoD = 0, ThrD = 1}; 

}

namespace Gaukuk{

template<class RiemannSolver, class Reconstruct, int DMS>
__global__
void UpdateCons2CoefKernel(const CArrayView<Real> deviConsView, 
                           const CArrayView<Real> deviPrimView, 
                           CArrayView<Real> consTempView, const EquationOfState eos, 
                           const Real coef1, const Real coef2, 
                           Real dtdx, Real dtdy, Real dtdz, 
                           int il, int ir, int jl, int jr, int kl, int kr){
    int i = il + blockIdx.x * blockDim.x + threadIdx.x;
    int j = jl + blockIdx.y * blockDim.y + threadIdx.y;
    int k = kl + blockIdx.z * blockDim.z + threadIdx.z;
    if (i>=il && i<ir && j>=jl && j<jr && k>=kl && k<kr){
        const Real dmin = DENSITY_FLOOR;
        const Real pmin = PRESSURE_FLOOR;
        const Real gm1 = eos.GetGamma() - Real(1.0); 
        Real ul[NVar]; 
        Real ur[NVar]; 
        Real fl[NVar]; 
        Real fr[NVar]; 
        Real Res[NVar]; 

        // load initial cons
        #pragma unroll
        for (int n=0; n<NVar; n++){
            Res[n] = coef1*deviConsView(n, k, j, i);
        }

        // flux on x
        // left side, between i-1 and i
        Reconstruct::ReconstructX(deviPrimView, ul, ur, i-1, j, k); 
        RiemannSolver::Solver(ul, ur, VLX, eos, fl); 
        // right side, between i and i+1
        Reconstruct::ReconstructX(deviPrimView, ul, ur, i, j, k); 
        RiemannSolver::Solver(ul, ur, VLX, eos, fr); 
        #pragma unroll
        for (int n=0; n<NVar; n++){
            Res[n] -= coef2*(fr[n] - fl[n])*dtdx;
        }

        // flux on y
        // left side, between j-1 and j
        Reconstruct::ReconstructY(deviPrimView, ul, ur, i, j-1, k); 
        RiemannSolver::Solver(ul, ur, VLY, eos, fl); 
        // right side, between j and j+1
        Reconstruct::ReconstructY(deviPrimView, ul, ur, i, j, k); 
        RiemannSolver::Solver(ul, ur, VLY, eos, fr); 
        #pragma unroll
        for (int n=0; n<NVar; n++){
            Res[n] -= coef2*(fr[n] - fl[n])*dtdy;
        }

        // flux on z
        if constexpr (DMS==ThrD){
            // left side, between k-1 and k
            Reconstruct::ReconstructZ(deviPrimView, ul, ur, i, j, k-1); 
            RiemannSolver::Solver(ul, ur, VLZ, eos, fl); 
            // right side, between k and k+1
            Reconstruct::ReconstructZ(deviPrimView, ul, ur, i, j, k); 
            RiemannSolver::Solver(ul, ur, VLZ, eos, fr); 
            #pragma unroll
            for (int n=0; n<NVar; n++){
                Res[n] -= coef2*(fr[n] - fl[n])*dtdz;
            }
        }

        if (Res[DEN] < dmin) {
            Res[DEN] = dmin;
            Res[MTX] = Real(0.0);
            Res[MTY] = Real(0.0);
            Res[MTZ] = Real(0.0);
            Res[ENG] = pmin / gm1 + Real(0.0);   // ke = 0
        } else {
            Real inv_den = Real(1.0) / Res[DEN];
            Real ke = Real(0.5) * inv_den * (Res[MTX]*Res[MTX] + Res[MTY]*Res[MTY] + Res[MTZ]*Res[MTZ]);
            Real press = gm1 * (Res[ENG] - ke);
            if (press < pmin) {
                Res[ENG] = pmin / gm1 + ke;   
            }
        }
        #pragma unroll
        for (int n=0; n<NVar; n++){
            consTempView(n, k, j, i) = Res[n];
        }
    }        
}

void Sim::UpdateCons(const CArray<Real>& deviCons, const CArray<Real>& deviPrim, 
                     CArray<Real>& consTemp, const EquationOfState& eos, 
                     const Real coef1, const Real coef2){
    int il = grid.ib; 
    int ir = grid.ie; 
    int jl = grid.jb; 
    int jr = grid.je; 
    int kl = grid.kb;
    int kr = grid.ke; 

    Real dtdx = dt*domain.dxRec; 
    Real dtdy = dt*domain.dyRec; 
    Real dtdz = dt*domain.dzRec; 

    dim3 cudaBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 cudaGrid(
        (ir - il + cudaBlock.x - 1) / cudaBlock.x,
        (jr - jl + cudaBlock.y - 1) / cudaBlock.y,
        (kr - kl + cudaBlock.z - 1) / cudaBlock.z
    );
    if (grid.nz == 1){
        UpdateCons2CoefKernel<RiemannSolverType, ReconstructType, TwoD><<<cudaGrid, cudaBlock>>>(
            deviCons.View(), deviPrim.View(), consTemp.View(), eos, coef1, coef2, 
            dtdx, dtdy, dtdz, il, ir, jl, jr, kl, kr
        );
    }else{
        UpdateCons2CoefKernel<RiemannSolverType, ReconstructType, ThrD><<<cudaGrid, cudaBlock>>>(
            deviCons.View(), deviPrim.View(), consTemp.View(), eos, coef1, coef2, 
            dtdx, dtdy, dtdz, il, ir, jl, jr, kl, kr
        );
    }
}

template<class RiemannSolver, class Reconstruct, int DMS>
__global__
void UpdateCons3CoefKernel(const CArrayView<Real> deviConsView, 
                           const CArrayView<Real> deviPrimView, 
                           CArrayView<Real> consTempView, const EquationOfState eos, 
                           const Real coef1, const Real coef2, const Real coef3, 
                           Real dtdx, Real dtdy, Real dtdz, 
                           int il, int ir, int jl, int jr, int kl, int kr){
    int i = il + blockIdx.x * blockDim.x + threadIdx.x;
    int j = jl + blockIdx.y * blockDim.y + threadIdx.y;
    int k = kl + blockIdx.z * blockDim.z + threadIdx.z;
    if (i>=il && i<ir && j>=jl && j<jr && k>=kl && k<kr){
        const Real dmin = DENSITY_FLOOR;
        const Real pmin = PRESSURE_FLOOR;
        const Real gm1 = eos.GetGamma() - Real(1.0); 
        Real ul[NVar]; 
        Real ur[NVar]; 
        Real fl[NVar]; 
        Real fr[NVar]; 
        Real Res[NVar]; 

        // load initial cons
        #pragma unroll
        for (int n=0; n<NVar; n++){
            Res[n] = coef1*deviConsView(n, k, j, i) + coef2*consTempView(n, k, j, i);
        }

        // flux on x
        // left side, between i-1 and i
        Reconstruct::ReconstructX(deviPrimView, ul, ur, i-1, j, k); 
        RiemannSolver::Solver(ul, ur, VLX, eos, fl); 
        // right side, between i and i+1
        Reconstruct::ReconstructX(deviPrimView, ul, ur, i, j, k); 
        RiemannSolver::Solver(ul, ur, VLX, eos, fr); 
        #pragma unroll
        for (int n=0; n<NVar; n++){
            Res[n] -= coef3*(fr[n] - fl[n])*dtdx;
        }

        // flux on y
        // left side, between j-1 and j
        Reconstruct::ReconstructY(deviPrimView, ul, ur, i, j-1, k); 
        RiemannSolver::Solver(ul, ur, VLY, eos, fl); 
        // right side, between j and j+1
        Reconstruct::ReconstructY(deviPrimView, ul, ur, i, j, k); 
        RiemannSolver::Solver(ul, ur, VLY, eos, fr); 
        #pragma unroll
        for (int n=0; n<NVar; n++){
            Res[n] -= coef3*(fr[n] - fl[n])*dtdy;
        }

        // flux on z
        if constexpr (DMS==ThrD){
            // left side, between k-1 and k
            Reconstruct::ReconstructZ(deviPrimView, ul, ur, i, j, k-1); 
            RiemannSolver::Solver(ul, ur, VLZ, eos, fl); 
            // right side, between k and k+1
            Reconstruct::ReconstructZ(deviPrimView, ul, ur, i, j, k); 
            RiemannSolver::Solver(ul, ur, VLZ, eos, fr); 
            #pragma unroll
            for (int n=0; n<NVar; n++){
                Res[n] -= coef3*(fr[n] - fl[n])*dtdz;
            }
        }

        if (Res[DEN] < dmin) {
            Res[DEN] = dmin;
            Res[MTX] = Real(0.0);
            Res[MTY] = Real(0.0);
            Res[MTZ] = Real(0.0);
            Res[ENG] = pmin / gm1 + Real(0.0);   // ke = 0
        } else {
            Real inv_den = Real(1.0) / Res[DEN];
            Real ke = Real(0.5) * inv_den * (Res[MTX]*Res[MTX] + Res[MTY]*Res[MTY] + Res[MTZ]*Res[MTZ]);
            Real press = gm1 * (Res[ENG] - ke);
            if (press < pmin) {
                Res[ENG] = pmin / gm1 + ke;   
            }
        }
        #pragma unroll
        for (int n=0; n<NVar; n++){
            consTempView(n, k, j, i) = Res[n];
        }
    }        
}

void Sim::UpdateCons(const CArray<Real>& deviCons, const CArray<Real>& deviPrim, 
                     CArray<Real>& consTemp, const EquationOfState& eos, 
                     const Real coef1, const Real coef2, const Real coef3){
    int il = grid.ib; 
    int ir = grid.ie; 
    int jl = grid.jb; 
    int jr = grid.je; 
    int kl = grid.kb;
    int kr = grid.ke; 

    Real dtdx = dt*domain.dxRec; 
    Real dtdy = dt*domain.dyRec; 
    Real dtdz = dt*domain.dzRec; 

    dim3 cudaBlock(BLOCK_X, BLOCK_Y, BLOCK_Z);
    dim3 cudaGrid(
        (ir - il + cudaBlock.x - 1) / cudaBlock.x,
        (jr - jl + cudaBlock.y - 1) / cudaBlock.y,
        (kr - kl + cudaBlock.z - 1) / cudaBlock.z
    );
    if (grid.nz == 1){
        UpdateCons3CoefKernel<RiemannSolverType, ReconstructType, TwoD><<<cudaGrid, cudaBlock>>>(
            deviCons.View(), deviPrim.View(), consTemp.View(), eos, coef1, coef2, coef3, 
            dtdx, dtdy, dtdz, il, ir, jl, jr, kl, kr
        );
    }else{
        UpdateCons3CoefKernel<RiemannSolverType, ReconstructType, ThrD><<<cudaGrid, cudaBlock>>>(
            deviCons.View(), deviPrim.View(), consTemp.View(), eos, coef1, coef2, coef3, 
            dtdx, dtdy, dtdz, il, ir, jl, jr, kl, kr
        );
    }
}

}
