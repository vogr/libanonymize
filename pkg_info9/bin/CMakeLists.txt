cmake_minimum_required(VERSION 3.13)
project(info9 VERSION 1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_PREFIX_PATH, ${CONDA_PREFIX})

list(APPEND ClassificationSource
  "../src/classification/ConfusionMatrix.cpp"
  "../src/classification/Dataset.cpp"
  "../src/classification/KnnClassification.cpp"
  "../src/classification/RandomProjection.cpp"
)

add_executable(info9-pgm main.cpp ../src/helpers.cpp ../src/classification ${ClassificationSource})

# Use OpenMP to parallelize some of Eigen's algorithms.
find_package(OpenMP REQUIRED)
if(OpenMP_CXX_FOUND)
  # This will generate a warning in conda environment
  target_link_libraries(info9-pgm PRIVATE OpenMP::OpenMP_CXX)
endif()

find_package(Eigen3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIRS})

set(HIGHFIVE_EXAMPLES FALSE)
set(HIGHFIVE_USE_BOOST FALSE)
set(HIGHFIVE_USE_EIGEN TRUE)
find_package(HighFive REQUIRED)
set(HIGHFIVE_USE_BOOST FALSE)
target_link_libraries(info9-pgm PRIVATE HighFive)

find_library(ANN NAMES ann)
target_link_libraries(info9-pgm PRIVATE ${ANN})

target_compile_options(info9-pgm
        PUBLIC -pipe
        $<$<CONFIG:Debug>:-O0 -ggdb3 -gsplit-dwarf>
        $<$<CONFIG:Release>:-flto -fno-fat-lto-objects> # release: 03 is implied
        $<$<CONFIG:RelWithDebInfo>:-ggdb3 -flto -fno-fat-lto-objects> # relwithdebinfo: 02 is implied
        -Wp,-D_FORTIFY_SOURCE=2  -Wp,-D_GLIBCXX_ASSERTIONS
        -fexceptions -fasynchronous-unwind-tables -fstack-protector-strong -fstack-clash-protection -fcf-protection
        -fPIE -Wl,-z,noexecstack,-z,relro,-z,defs,-pie,-z,now
        PRIVATE -Wall -Wextra -Wpedantic -Wformat=2 -Wswitch-default -Wswitch-enum -Wfloat-equal -Wno-conversion
        -pedantic-errors -Werror=format-security
    )
