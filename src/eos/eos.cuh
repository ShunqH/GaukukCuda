#pragma once 

// C++ headers
#include <cmath>            // sqrt()

// Gaukuk dependence
#include "../gaukuk.hpp"
#include "../template_array.hpp"
#include "../cuda_array.cuh"
#include "../cuda_value.cuh"
#include "../grid.hpp"

namespace Gaukuk{

class EquationOfState{
public:
    EquationOfState(); 
    void ConsToPrim(const CArray<Real>& deviCons, CArray<Real>& deviPrim, 
                    const Grid& grid); 
    void CalCmax(const CArray<Real>& deviCons, const Grid& grid, CValue<Real>& deviCmax);  
    void PrimToCons(const CArray<Real>& deviPrim, CArray<Real>& deviCons, 
                    const Grid& grid); 
    
    __host__ __device__
    inline Real GetGamma() const { return gamma_; }

    __host__ __device__
    inline Real GetGm1Rec() const { return gm1Rec_; }
    
    __host__ __device__ __forceinline__
    Real SoundSpeed(const Real den, const Real pre) const {
        return std::sqrt(gamma_*pre/den); 
    }
    __host__ __device__ __forceinline__
    Real EGas(const Real den, const Real pre) const {
        return pre * gm1Rec_; 
    }
    
private:
    Real gamma_, gm1Rec_; 
};


} // namespace Gaukuk