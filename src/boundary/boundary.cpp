// C++ Headers
#include <iostream>     // std::cout; std::endl; std::cerr

// Gaukuk dependence
#include "boundary.hpp"

namespace Gaukuk
{

Boundary::Boundary() {
    int typeBDXL = static_cast<int>(Config::getInstance().get("xleft")); 
    int typeBDXR = static_cast<int>(Config::getInstance().get("xright")); 
    int typeBDYL = static_cast<int>(Config::getInstance().get("yleft")); 
    int typeBDYR = static_cast<int>(Config::getInstance().get("yright")); 
    int typeBDZL = static_cast<int>(Config::getInstance().get("zleft")); 
    int typeBDZR = static_cast<int>(Config::getInstance().get("zright"));

    // X left boundary registration
    if (typeBDXL == 0){
        Bdxl = &OutflowCopyXL;
    }else if (typeBDXL == 1){
        Bdxl = &PeriodicXL;
    }

    // X right boundary registration
    if (typeBDXR == 0){
        Bdxr = &OutflowCopyXR;
    }else if (typeBDXR == 1){
        Bdxr = &PeriodicXR;
    }

    // y left boundary registration
    if (typeBDYL == 0){
        Bdyl = &OutflowCopyYL;
    }else if (typeBDYL == 1){
        Bdyl = &PeriodicYL;
    }

    // y right boundary registration
    if (typeBDYR == 0){
        Bdyr = &OutflowCopyYR;
    }else if (typeBDYR == 1){
        Bdyr = &PeriodicYR;
    }

    // z left boundary registration
    if (typeBDZL == 0){
        Bdzl = &OutflowCopyYL;
    }else if (typeBDZL == 1){
        Bdzl = &PeriodicZL;
    }

    // z right boundary registration
    if (typeBDZR == 0){
        Bdzr = &OutflowCopyYR;
    }else if (typeBDZR == 1){
        Bdzr = &PeriodicZR;
    }
}

void Boundary::UpdateBD(CArray<Real>& deviCons, const Grid& grid, 
                        const Domain& domain, const EquationOfState& eos){
    Bdxl(deviCons, grid, eos); 
    Bdxr(deviCons, grid, eos); 
    Bdyl(deviCons, grid, eos); 
    Bdyr(deviCons, grid, eos); 
if (grid.nz>1){
    Bdzl(deviCons, grid, eos); 
    Bdzr(deviCons, grid, eos); 
}
}

} // namespace Gaukuk