#pragma once 

// C++ headers
#include <functional>

// Gaukuk dependence
#include "../gaukuk.hpp" 
#include "../cuda_array.cuh"
#include "../eos/eos.cuh"

namespace Gaukuk{

/* 
---------------------------------------------------------------
    HLLC Riemann Solver 
    Toro E.F.
    Riemann Solvers and Numerical Methods for Fluid Dynamics 
    Chapter 10.4 ~ 10.6
----------------------------------------------------------------
*/

struct HLLC{
public:
    static __device__ __forceinline__
    void Solver(Real ul[NVar], Real ur[NVar], const int direction, 
                const EquationOfState& eos, Real flux[NVar]){
        const int IVLX = VLX + (direction - VLX + 0) % 3; 
        const int IVLY = VLX + (direction - VLX + 1) % 3; 
        const int IVLZ = VLX + (direction - VLX + 2) % 3; 
        Real gamma = eos.GetGamma(); 
        Real gmRec = Real(1.0)/gamma; 

        // load data
        Real denl = ul[DEN]; 
        Real vxl  = ul[IVLX]; 
        Real vyl  = ul[IVLY]; 
        Real vzl  = ul[IVLZ]; 
        Real prel = ul[PRE];

        Real denr = ur[DEN]; 
        Real vxr  = ur[IVLX]; 
        Real vyr  = ur[IVLY]; 
        Real vzr  = ur[IVLZ]; 
        Real prer = ur[PRE];

        // Step I pressure estimate. 
        Real al = eos.SoundSpeed(denl, prel); 
        Real ar = eos.SoundSpeed(denr, prer); 
        Real prePVRS = Real(0.5)*(prel + prer) - Real(0.125)*(vxr - vxl)*(denl + denr)*(al + ar); 
        prePVRS = fmax(prePVRS, Real(0.0));

        // Step II wave speed estimates. 
        // ql = (prePVRS<=prel) ? Real(1.0) : sqrt( Real(1.0) + 0.5*(Real(1.0) + gmRec) * (prePVRS/prel - Real(1.0)) ); 
        // qr = (prePVRS<=prer) ? Real(1.0) : sqrt( Real(1.0) + 0.5*(Real(1.0) + gmRec) * (prePVRS/prer - Real(1.0)) ); 
        Real temp = fmax(prePVRS/prel-Real(1.0), Real(0.0)); 
        Real ql = sqrt( Real(1.0) + Real(0.5)*(Real(1.0) + gmRec) * temp );
        temp = fmax(prePVRS/prer-Real(1.0), Real(0)); 
        Real qr = sqrt( Real(1.0) + Real(0.5)*(Real(1.0) + gmRec) * temp );

        Real sl = vxl - al*ql; 
        Real sr = vxr + ar*qr; 
        Real ss = ( prer - prel + denl*vxl*(sl-vxl) - denr*vxr*(sr-vxr) ) / 
                  ( denl*(sl-vxl) - denr*(sr-vxr) ) ;

        // Step III HLLC flux
        Real el = eos.EGas(denl, prel) + Real(0.5)*denl*(vxl*vxl+vyl*vyl+vzl*vzl); 
        Real er = eos.EGas(denr, prer) + Real(0.5)*denr*(vxr*vxr+vyr*vyr+vzr*vzr); 

        Real cl = fmin(Real(0.0), sl)*(ss-vxl)/(sl-ss); 
        Real cr = fmax(Real(0.0), sr)*(ss-vxr)/(sr-ss); 
        Real selectl = (ss >= Real(0.0)) ? Real(1.0) : Real(0.0);
        Real selectr = Real(1.0) - selectl;

        Real fl1 = denl * (vxl + cl);  
        Real fl2 = denl * (vxl*vxl + cl*sl) + prel; 
        Real fl3 = fl1*vyl;                             // denl*vxl*vyl     + denl*cl*vyl;
        Real fl4 = fl1*vzl;                             // denl*vxl*vzl     + denl*cl*vzl; 
        Real fl5 = (el + prel)*(vxl + cl) + denl*cl*ss*(sl-vxl);

        Real fr1 = denr * (vxr + cr);  
        Real fr2 = denr * (vxr*vxr + cr*sr) + prer; 
        Real fr3 = fr1*vyr;                             // denr*vxr*vyr     + denr*cr*vyr;
        Real fr4 = fr1*vzr;                             // denr*vxr*vzr     + denr*cr*vzr; 
        Real fr5 = (er + prer)*(vxr + cr) + denr*cr*ss*(sr-vxr);

        flux[DEN]  = selectl * fl1 + selectr * fr1; 
        flux[IVLX] = selectl * fl2 + selectr * fr2; 
        flux[IVLY] = selectl * fl3 + selectr * fr3; 
        flux[IVLZ] = selectl * fl4 + selectr * fr4; 
        flux[ENG]  = selectl * fl5 + selectr * fr5; 
    }
};

}