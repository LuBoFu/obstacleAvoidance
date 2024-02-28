cmake_minimum_required(VERSION 3.1)
list(APPEND CMAKE_MODULE_PATH "/s/cmake/zb-cmake-lib")
include(includeAll)
include(packagesForAsus3)

add_compile_options(--std=c++17)

createSimpleApplication("generateKittiRawIndex"  "src"  
    "t_options;t_h5pp;t_fileIOHelper"  "bin")

