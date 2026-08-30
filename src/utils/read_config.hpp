#pragma once 

// C++ headers
#include <map>
#include <string> 

// Gaukuk header
#include "../gaukuk.hpp"

namespace Gaukuk{

class Config {
public:
    static Config& getInstance();
    void loadFromFile(const std::string& filename);
    Real get(const std::string& key, Real defaultValue = 0.0);
    std::string getString(const std::string& key, const std::string& defaultValue = "");

private:
    Config();  
    std::map<std::string, std::string> data;
    static std::string trim(const std::string& str);
};

namespace GMath {
    inline Real SQR(Real a){ return a*a; }
}; 

} // namespace Gaukuk