#pragma once 

// C++ headers
#include <functional>

// Gaukuk dependence
#include "../gaukuk.hpp" 
#include "../template_array.hpp" 
#include "../cuda_array.cuh"
#include "../grid.hpp"
#include "../eos/eos.cuh"

namespace Gaukuk
{

class Boundary{
// using BoundaryFunc = void (*)(CArray<Real>&, const Grid&, const EquationOfState& eos);
using BoundaryFunc = std::function<void(CArray<Real>&, const Grid&, 
                                        const EquationOfState&)>;
public:
// friend class Sim; 
    Boundary(); 
    BoundaryFunc Bdxl; 
    BoundaryFunc Bdxr; 
    BoundaryFunc Bdyl; 
    BoundaryFunc Bdyr; 
    BoundaryFunc Bdzl; 
    BoundaryFunc Bdzr; 
    void UpdateBD(CArray<Real>& deviCons, 
                  const Grid& grid, const Domain& domain, 
                  const EquationOfState& eos); 

private:
    // simple copy boundary condition
    static void OutflowCopyXL(CArray<Real>& deviCons, const Grid& grid, const EquationOfState& eos); 
    static void OutflowCopyXR(CArray<Real>& deviCons, const Grid& grid, const EquationOfState& eos); 
    static void OutflowCopyYL(CArray<Real>& deviCons, const Grid& grid, const EquationOfState& eos); 
    static void OutflowCopyYR(CArray<Real>& deviCons, const Grid& grid, const EquationOfState& eos); 
    static void OutflowCopyZL(CArray<Real>& deviCons, const Grid& grid, const EquationOfState& eos); 
    static void OutflowCopyZR(CArray<Real>& deviCons, const Grid& grid, const EquationOfState& eos); 

    // periodic boundary condition
    static void PeriodicXL(CArray<Real>& deviCons, const Grid& grid, const EquationOfState& eos); 
    static void PeriodicXR(CArray<Real>& deviCons, const Grid& grid, const EquationOfState& eos); 
    static void PeriodicYL(CArray<Real>& deviCons, const Grid& grid, const EquationOfState& eos); 
    static void PeriodicYR(CArray<Real>& deviCons, const Grid& grid, const EquationOfState& eos); 
    static void PeriodicZL(CArray<Real>& deviCons, const Grid& grid, const EquationOfState& eos); 
    static void PeriodicZR(CArray<Real>& deviCons, const Grid& grid, const EquationOfState& eos); 
}; 

}