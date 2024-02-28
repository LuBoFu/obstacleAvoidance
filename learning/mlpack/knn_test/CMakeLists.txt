cmake_minimum_required(VERSION 3.1)
list(APPEND CMAKE_MODULE_PATH "/s/cmake/zb-cmake-lib")
include(includeAll)
include(packagesForAsus3)

createSimpleApplication("knn_test"  "src"
    "t_catch2;armadillo"
    "bin")

