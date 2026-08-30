// CPP header
#include <algorithm> 
#include <iostream>     // std::cout; std::endl; std::cerr
#include <chrono>       // std::chrono::high_resolution_clock
#include <ctime>        // clock_t
#include <iomanip>      // std::setw()

// Gaukuk dependence
#include "gaukuk.hpp" 
#include "sim.hpp" 

int main(int argc, char* argv[]){
    using namespace Gaukuk; 

    if (argc < 3 || std::string(argv[1]) != "-i") {
        std::cerr << "Usage: " << argv[0] << " -i input.in" << std::endl;
        return 1;
    }
    Config::getInstance().loadFromFile(argv[2]);

    Real tmax = Config::getInstance().get("tmax"); 
    Real dtoutput = Config::getInstance().get("dtoutput"); 

    DataType outputType = DataType::Cons; 
    int oTypeFromConfig = static_cast<int>(Config::getInstance().get("dataType")); 
    if (oTypeFromConfig == 0){
        outputType = DataType::Cons; 
    }else if (oTypeFromConfig == 1){
        outputType = DataType::Prim; 
    }

    Sim sim; 
    sim.Setup(); 
    int outputStep = 0; 
    Real tnow = sim.GetTime(); 

    sim.deviCons.CopyFromHost(sim.hostCons); 
    sim.boundary.UpdateBD(sim.deviCons, sim.grid, sim.domain, sim.eos);
    sim.deviCons.CopyToHost(sim.hostCons); 
    sim.WriteData(outputStep, outputType); 

    auto time0 = std::chrono::high_resolution_clock::now();
    clock_t cputime0 = clock();
    while (std::abs(tmax - tnow) > 1e-12 && sim.isContinue){
        dtoutput = std::min(dtoutput, tmax-tnow); 
        sim.Advance(dtoutput); 
        tnow = sim.GetTime();
        outputStep ++;  
        sim.WriteData(outputStep, outputType); 
        std::cout<<"output "<< outputStep << std::endl; 
        // WriteTarray(sim.flx1, "flx1", outputStep); 
    }

    auto time1 = std::chrono::high_resolution_clock::now();
    clock_t cputime1 = clock();
    // sim.WriteData(outputStep, outputType); 
    Real walltime = std::chrono::duration<Real>(time1 - time0).count(); 
    Real cputime = double(cputime1 - cputime0) / CLOCKS_PER_SEC; 

    std::cout << std::right
            << "Finish! " 
            << "\n    total steps = " << std::setw(6) << sim.GetStep()
            << " , \n    walltime = " << std::setw(8) << std::fixed << std::setprecision(3) << walltime
            << " s, \n    cputime = " << std::setw(8) << std::fixed << std::setprecision(3) << cputime << " s"
            << "\n-----------------------------------------------------------"
            << std::endl;
}