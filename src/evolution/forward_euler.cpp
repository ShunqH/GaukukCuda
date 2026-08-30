// Gaukuk dependence
#include "../gaukuk.hpp"
// #include "../cuda_array.cuh"
#include "../sim.hpp" 

namespace Gaukuk
{

void Sim::ForwardEuler_(){
    boundary.UpdateBD(deviCons, grid, domain, eos); 
    eos.ConsToPrim(deviCons, deviPrim, grid); 
    UpdateCons(deviCons, deviPrim, consTemp, eos, 1.0, 1.0); 
    deviCons.Swap(consTemp); 
}

} // namespace Gaukuk