#pragma once 

namespace Gaukuk{

#if defined(GAUKUK_USE_FLOAT)
    using Real = float;
#else
    using Real = double;
#endif

#if defined(GAUKUK_DENSITY_FLOOR)
#define GAUKUK_DENSITY_FLOOR 1e-16
#endif
#if defined(GAUKUK_PRESSURE_FLOOR)
#define GAUKUK_PRESSURE_FLOOR 1e-16
#endif

struct HLLC;
struct ROE;
#if defined(USE_ROE)
    using RiemannSolverType = ROE;
#else 
    using RiemannSolverType = HLLC;
#endif

struct RCFirstOrder;
struct RCPLM;
#if defined(USE_FIRST_ORDER)
    using ReconstructType = RCFirstOrder;
    constexpr int MIN_GHOSTS = 1;
#else 
    using ReconstructType = RCPLM;
    constexpr int MIN_GHOSTS = 2;
#endif

constexpr int NVar = 5;
constexpr Real PI = 3.1415926535; 
constexpr Real DENSITY_FLOOR  = GAUKUK_DENSITY_FLOOR;
constexpr Real PRESSURE_FLOOR = GAUKUK_PRESSURE_FLOOR;
constexpr Real CMAX_FLOOR = 1e-16;

enum ConsIDs {DEN = 0, MTX = 1, MTY = 2, MTZ = 3, ENG = 4, }; 
enum PrimIDs {VLX = 1, VLY = 2, VLZ = 3, PRE = 4, }; 

inline constexpr int BLOCK_X = 32;
inline constexpr int BLOCK_Y = 4;
inline constexpr int BLOCK_Z = 1;
inline constexpr int BLOCK_SIZE = BLOCK_X*BLOCK_Y*BLOCK_Z; 

} // namespace Gaukuk