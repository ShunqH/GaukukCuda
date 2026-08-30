// CPP header
#include <stdexcept>    // runtime_error
#include <algorithm>    // std::min()
#include <iostream>     // std::cout; std::endl; std::cerr

// Gaukuk dependence
#include "sim.hpp"
#include "utils/read_config.hpp"

namespace Gaukuk{

Domain::Domain(const Grid& grid) : 
    xmin(Config::getInstance().get("xmin")), 
    xmax(Config::getInstance().get("xmax")), 
    ymin(Config::getInstance().get("ymin")), 
    ymax(Config::getInstance().get("ymax")), 
    zmin(Config::getInstance().get("zmin")), 
    zmax(Config::getInstance().get("zmax")) {
    dx = ( xmax - xmin ) / grid.nx; 
    dy = ( ymax - ymin ) / grid.ny; 
    dz = ( zmax - zmin ) / grid.nz; 
    dxRec = 1./dx; 
    dyRec = 1./dy; 
    dzRec = 1./dz; 
    if (grid.ny<=1){
        dyRec = 0; 
    }
    if (grid.nz<=1){
        dzRec = 0; 
    }
    drmin = std::min(std::min(dx, dy), dz); 
    xc.NewArray(grid.lenx); 
    yc.NewArray(grid.leny); 
    zc.NewArray(grid.lenz); 
    for (int i=0; i<grid.lenx; i++) { xc(i) = xmin - grid.nGhost*dx + 0.5*dx + i*dx; }
    for (int j=0; j<grid.leny; j++) { yc(j) = ymin - grid.nGhost*dy + 0.5*dy + j*dy; }
    for (int k=0; k<grid.lenz; k++) { zc(k) = zmin - grid.nGhost*dz + 0.5*dz + k*dz; }
}

Sim::Sim(): isContinue(true), domain(grid), //flux(grid.lenx), 
            step(0), t(0), dt(1e10), dtUntilOutput(1e10), cmax(1e-16), deviCmax(1e-16){
    CFL = Config::getInstance().get("CFL"); 

    if (grid.nGhost < Gaukuk::MIN_GHOSTS) {
        std::cerr << "========================================================\n"
                  << "[FATAL ERROR] Ghost cell mismatch!\n"
                  << "The compiled reconstruction method requires at least " 
                  << Gaukuk::MIN_GHOSTS << " ghost cells.\n"
                  << "Please increase NGhost in the input file and try again.\n"
                  << "========================================================\n";
        std::exit(EXIT_FAILURE); 
    }

    hostCons.NewArray(NVar, grid.lenz, grid.leny, grid.lenx);
    deviCons.NewArray(NVar, grid.lenz, grid.leny, grid.lenx);
    deviPrim.NewArray(NVar, grid.lenz, grid.leny, grid.lenx);
    consTemp.NewArray(NVar, grid.lenz, grid.leny, grid.lenx);
    
    integratorType = static_cast<int>(Config::getInstance().get("integrator", 2)); 
    stepMax = Config::getInstance().get("stepmax", -1); 
    
    if (integratorType == 1) {
        hydroIntegrator_ = &Sim::ForwardEuler_; 
    }else if (integratorType == 3) {
        hydroIntegrator_ = &Sim::RK3_; 
    }else {
        hydroIntegrator_ = &Sim::RK2_; 
    }
}

void Sim::Advance(Real dtoutput){
    Real tNext = t + dtoutput; 
    dtUntilOutput = dtoutput; 

    while (std::abs(tNext - t) > 1e-12 && isContinue){

        // auto time0 = std::chrono::high_resolution_clock::now();
        // clock_t cputime0 = clock(); 

        cmax = 1e-16;
        eos.CalCmax(deviCons, grid, deviCmax); 
        deviCmax.CopyToHost(cmax); 
        dt = CFL * domain.drmin / cmax; 
        dt = std::min(dt, dtUntilOutput); 

        // Strang Splitting 
        // source(0.5*dt) -> hydro(dt) -> source(0.5*dt)
        // if (srcTerm.sourceEnrolled){
        //     srcTerm.UpdateSource(cons, t+0.5*dt, 0.5*dt, grid, domain); 
        // }
        (this->*hydroIntegrator_)(); 
        // if (srcTerm.sourceEnrolled){
        //     srcTerm.UpdateSource(cons, t + dt, 0.5*dt, grid, domain); 
        // }
        // WriteData(step, DataType::Cons);
        t += dt; 
        step ++; 
        dtUntilOutput = tNext - t; 
        if (stepMax>0 && step>=stepMax){
            isContinue = false; 
        }

        std::cout << "step = " << step 
                  << ", t = " << t 
                  << ", dt = " << dt 
                  << std::endl; 
    }
}

}