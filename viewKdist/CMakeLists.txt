cmake_minimum_required(VERSION 3.1)
list(APPEND CMAKE_MODULE_PATH "/s/cmake/zb-cmake-lib")
include(includeAll)
include(packagesForAsus3)

createSimpleApplication("viewKdist"  "src"
    "t_options;t_h5pp;armadillo;t_matplotpp"
    "bin")

