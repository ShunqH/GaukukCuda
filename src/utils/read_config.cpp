#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <cctype>  // for std::isspace

#include "read_config.hpp"

namespace Gaukuk{

Config& Config::getInstance() {
    static Config instance;
    return instance;
}

Config::Config() {}

void Config::loadFromFile(const std::string& filename) {
    using namespace std; 
    ifstream file(filename);
    if (!file) {
        throw invalid_argument("Could not open the input file");
    }

    string line;
    while (getline(file, line)) {
        line = trim(line);
        if (line.empty() || line[0] == '#') continue;  // Skip comments and empty lines

        // Remove '=' if it exists and split by space or '='
        size_t pos = line.find('=');
        if (pos != string::npos) {
            line.replace(pos, 1, " "); // Replace '=' with a space
        }

        istringstream iss(line);
        string key;
        if (iss >> key) {
            string value; 
            iss >> ws;
            getline(iss, value); 
            data[key] = trim(value);
        } else {
            cout << key << endl; 
            cerr << "Warning: Invalid line in config file: " << line << endl;
        }
    }
}

Real Config::get(const std::string& key, Real defaultValue) {
    if (data.count(key)) {
        try {
            return std::stod(data[key]); 
        } catch (...) {
            std::cerr << "Warning: Could not convert key '" << key << "' to Real. Using default." << std::endl;
            return defaultValue;
        }
    }
    return defaultValue;
}

std::string Config::getString(const std::string& key, const std::string& defaultValue) {
    return data.count(key) ? data[key] : defaultValue;
}

std::string Config::trim(const std::string& str) {
    auto first = str.begin();
    while (first != str.end() && std::isspace(static_cast<unsigned char>(*first))) {
        ++first;
    }
    if (first == str.end()) return "";
    auto last = str.end() - 1;
    while (last > first && std::isspace(static_cast<unsigned char>(*last))) {
        --last;
    }
    return std::string(first, last + 1);
}

} // namespace Gaukuk