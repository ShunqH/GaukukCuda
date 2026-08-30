// C++ headers
#include <cmath>        //std::sin

// Gaukuk dependence
#include "../gaukuk.hpp" 
#include "../sim.hpp" 

namespace Gaukuk
{
    
void Sim::Setup(){
    // load from setupfile 
    Real rhoIn = Config::getInstance().get("rhoIn"); 
    Real rhoOut = Config::getInstance().get("rhoOut"); 
    Real vxIn = Config::getInstance().get("vxIn"); 
    Real vxOut = Config::getInstance().get("vxOut"); 
    Real pressure = Config::getInstance().get("pressure"); 
    Real amp = Config::getInstance().get("amp"); 
    Real gamma = Config::getInstance().get("gamma") ; 

    Real yLayer = domain.ymax/2; 
    Real xmin = domain.xmin; 
    Real Lx = domain.xmax - domain.xmin; 
    Real ymin = domain.ymin; 
    Real Ly = domain.ymax - domain.ymin; 
    Real gm1Rec = 1.0 / (gamma - 1.0);

    // activated zone
    int il = grid.ib; 
    int ir = grid.ie; 
    int jl = grid.jb; 
    int jr = grid.je; 
    int kl = grid.kb;
    int kr = grid.ke; 

#pragma omp parallel for collapse(2) schedule(static)    
    for (int k=kl; k<kr; k++){
        for (int j=jl; j<jr; j++){
#pragma omp simd
            for (int i=il; i<ir; i++){
                Real xNow = domain.xc(i); 
                Real yNow = domain.yc(j); 

                Real rhoNow = (std::abs(yNow)<yLayer) ? rhoIn : rhoOut ; 
                Real vx = (std::abs(yNow)<yLayer) ? vxIn : vxOut ; 
                Real vy = 0; 
                Real vz = 0; 

                Real vxRandom = amp * std::sin( 4 * PI * (xNow-xmin) / Lx ); 
                Real vyRandom = amp * std::cos( 4 * PI * (xNow-xmin) / Lx );  
                vx += vxRandom; 
                vy += vyRandom; 

                // usually you have to setup conservative quantivities (cons) 
                // but you can setup primitive quantivities (cons) 
                // then use eos.PrimToCons to conver 
                hostCons(DEN, k, j, i) = rhoNow; 
                hostCons(MTX, k, j, i) = rhoNow*vx; 
                hostCons(MTY, k, j, i) = rhoNow*vy; 
                hostCons(MTZ, k, j, i) = 0; 
                hostCons(PRE, k, j, i) = pressure*gm1Rec + 0.5 * rhoNow * ( vx*vx + vy*vy +vz*vz );
            }
        }
    }
    // if primitive quantivities is set, make sure you call eos.PrimToCons 
    // eos.PrimToCons(prim, cons, grid); 
}

} // namespace Gaukuk
