cmake_minimum_required(VERSION 3.13)
project(info9 VERSION 1.0)

set(CMAKE_PREFIX_PATH, ${CONDA_PREFIX})

#include_directories("src/")


find_package(OpenMP REQUIRED)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(HIGHFIVE_EXAMPLES FALSE)
set(HIGHFIVE_USE_BOOST FALSE)
set(HIGHFIVE_USE_EIGEN TRUE)
find_package(HighFive REQUIRED)

find_library(ANN NAMES ann)

# Define "info9-lib" library
add_subdirectory("src")


find_package(pybind11 REQUIRED)
#add_subdirectory("extern/pybind11")
pybind11_add_module(info9 MODULE "bindings/wrap.cpp")

target_link_libraries(info9 PRIVATE info9-lib)
target_link_libraries(info9 PRIVATE Eigen3::Eigen)
target_link_libraries(info9 PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(info9 PRIVATE HighFive)
target_link_libraries(info9 PRIVATE ${ANN})

target_compile_options(info9
      PUBLIC
      $<$<CONFIG:Debug>:-O0 -ggdb3 -gsplit-dwarf>
      $<$<CONFIG:Release>:-O2 -march=native -flto -fno-fat-lto-objects> # release: 03 is implied, add back 02
      $<$<CONFIG:RelWithDebInfo>:-O2 -march=native -ggdb3 -flto -fno-fat-lto-objects> # relwithdebinfo: 02 is implied
  )
