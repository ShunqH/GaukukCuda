#pragma once 

// C++ headers
#include <cstddef>  // size_t
#include <future>   // std::future

// Gaukuk dependence
#include "gaukuk.hpp" 
#include "template_array.hpp"
#include "cuda_array.cuh"
#include "cuda_value.cuh"
#include "grid.hpp"
#include "./eos/eos.cuh"
#include "boundary/boundary.hpp" 
// #include "source_term/source.hpp"

namespace Gaukuk{

enum class DataType {
    Prim,
    Cons
};

class Sim{
public:
    Sim(); 
    bool isContinue; 
    const Grid grid; 
    const Domain domain; 

    TArray<Real> hostCons;
    CArray<Real> deviCons, deviPrim;

    EquationOfState eos; 
    Boundary boundary; 
    // SourceTerm srcTerm; 
    
    void Setup(); 
    void Advance(Real dtoutput);
    
    void WriteData(const int outputID, DataType dType); 

    Real GetTime(){ return t; }
    Real Getdt(){ return dt; }
    int GetStep() { return step; }
private:
using VoidFunc = void (Sim::*)();
    int step, stepMax; 
    Real t, dt, dtUntilOutput, cmax, CFL; 
    CValue<Real> deviCmax; 
    int integratorType; 
    VoidFunc hydroIntegrator_; 
    void UpdateCons(const CArray<Real>& deviCons, const CArray<Real>& deviPrim, 
                    CArray<Real>& consTemp, const EquationOfState& eos,
                    const Real coef1, const Real coef2); 
    void UpdateCons(const CArray<Real>& deviCons, const CArray<Real>& deviPrim, 
                    CArray<Real>& consTemp, const EquationOfState& eos,
                    const Real coef1, const Real coef2, const Real coef3); 
    void ForwardEuler_(); 
    void RK2_(); 
    void RK3_(); 
    CArray<Real> consTemp; 

    std::future<void> writeFuture_;
}; 

} // namespace Gaukuk