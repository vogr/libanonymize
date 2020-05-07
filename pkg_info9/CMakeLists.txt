cmake_minimum_required(VERSION 3.11)
project(info9 VERSION 1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_PREFIX_PATH, ${CONDA_PREFIX})


find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIRS})

set(HIGHFIVE_EXAMPLES FALSE)
set(HIGHFIVE_USE_BOOST FALSE)
set(HIGHFIVE_USE_EIGEN TRUE)
find_package(HighFive REQUIRED)
#add_subdirectory(extern/HighFive)

#find_package(HDF5 REQUIRED)

find_library(ANN NAMES ann)

list(APPEND ClassificationSource
  "src/classification/Dataset.cpp"
  "src/classification/KnnClassification.cpp"
)

add_subdirectory(extern/pybind11)
pybind11_add_module(info9 src/wrap.cpp src/helpers.cpp src/classification ${ClassificationSource})
target_link_libraries(info9
  PRIVATE
  HighFive
  ${ANN}
)
