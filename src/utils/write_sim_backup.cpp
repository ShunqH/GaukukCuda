// C++ headers
#include <string>           // std::string 
#include <fstream>          // std::ofstream
#include <sstream>          // std::ostringstream; .str(); 
#include <iomanip>          // std::setw(); std::setfill()
#include <filesystem>

// Gaukuk dependence
#include "../gaukuk.hpp" 
#include "../sim.hpp" 

namespace Gaukuk
{
    
void Sim::WriteData(const int outputID, DataType dType){
    std::string outDir = Config::getInstance().getString("OutputDir", "./");
    if (!outDir.empty() && outDir.back() != '/') {
        outDir += '/';
    }
    // c++17
    if (!std::filesystem::exists(outDir)) {
        std::filesystem::create_directories(outDir);
    }

    std::ostringstream filename_stream;
    filename_stream << outDir;
    if (dType == DataType::Prim){
        filename_stream << "prim_" << std::setw(5) << std::setfill('0') << outputID;
    }else{
        filename_stream << "cons_" << std::setw(5) << std::setfill('0') << outputID;
    }
    std::string filename = filename_stream.str(); 

    std::ofstream outFile(filename, std::ios::binary); 
    if (!outFile) {
        throw std::runtime_error("Failed to open " + filename);
    }

    int real_size = sizeof(Real);
    
    outFile.write(reinterpret_cast<const char*>(&real_size), sizeof(int));

    // write frame information
    outFile.write(reinterpret_cast<const char*>(&t), sizeof(Real));

    // write number of quantity and size of x, y, z, and ghost
    outFile.write(reinterpret_cast<const char*>(&NVar), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&grid.nx), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&grid.ny), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&grid.nz), sizeof(int));
    outFile.write(reinterpret_cast<const char*>(&grid.nGhost), sizeof(int));

    // write mesh: x, y, z 
    outFile.write(reinterpret_cast<const char*>(domain.xc.data()), domain.xc.GetSizeInBytes());
    outFile.write(reinterpret_cast<const char*>(domain.yc.data()), domain.yc.GetSizeInBytes());
    outFile.write(reinterpret_cast<const char*>(domain.zc.data()), domain.zc.GetSizeInBytes());

    // write data
    if (dType == DataType::Prim){
        eos.ConsToPrim(deviCons, deviPrim, grid); 
        deviPrim.CopyToHost(hostCons); 
        int size = hostCons.GetSize();
        outFile.write(reinterpret_cast<const char*>(&size), sizeof(int));
        outFile.write(reinterpret_cast<const char*>(hostCons.data()), hostCons.GetSizeInBytes()); 
    }else{
        deviCons.CopyToHost(hostCons); 
        int size = hostCons.GetSize();
        outFile.write(reinterpret_cast<const char*>(&size), sizeof(int));
        outFile.write(reinterpret_cast<const char*>(hostCons.data()), hostCons.GetSizeInBytes()); 
    }
}

} // namespace Gaukuk
