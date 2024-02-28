cmake_minimum_required(VERSION 3.1)
list(APPEND CMAKE_MODULE_PATH "/s/cmake/zb-cmake-lib")
include(includeAll)
include(packagesForAsus3)

createSimpleApplication("personDetection"  "src"
    "t_options;t_h5pp;t_opencv;t_eigen"
    "bin")

