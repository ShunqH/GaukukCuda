// Gaukuk dependence
#include "../gaukuk.hpp"
// #include "../cuda_array.cuh"
#include "../sim.hpp" 

namespace Gaukuk
{

void Sim::RK3_(){
    boundary.UpdateBD(deviCons, grid, domain, eos); 
    eos.ConsToPrim(deviCons, deviPrim, grid); 
    UpdateCons(deviCons, deviPrim, consTemp, eos, 1.0, 1.0); 

    boundary.UpdateBD(consTemp, grid, domain, eos); 
    eos.ConsToPrim(consTemp, deviPrim, grid); 
    UpdateCons(deviCons, deviPrim, consTemp, eos, 0.75, 0.25, 0.25); 

    boundary.UpdateBD(consTemp, grid, domain, eos); 
    eos.ConsToPrim(consTemp, deviPrim, grid); 
    Real frac13 = 1.0/3.0; 
    UpdateCons(deviCons, deviPrim, consTemp, eos, frac13, 2*frac13, 2*frac13); 
    deviCons.Swap(consTemp); 
}

} // namespace Gaukuk