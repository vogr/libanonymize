cmake_minimum_required(VERSION 3.13)
project(logistic VERSION 1.0)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_PREFIX_PATH, ${CONDA_PREFIX})


find_package(OpenMP REQUIRED)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(HIGHFIVE_EXAMPLES FALSE)
set(HIGHFIVE_USE_BOOST FALSE)
set(HIGHFIVE_USE_EIGEN TRUE)
find_package(HighFive REQUIRED)

find_library(ANN NAMES ann)

# Define "info9-lib" library
add_subdirectory("../../pkg_info9/src" "libinfo9")

add_executable(logistic main.cpp)

target_link_libraries(logistic PRIVATE info9-lib)
target_link_libraries(logistic PRIVATE Eigen3::Eigen)
target_link_libraries(logistic PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(logistic PRIVATE HighFive)
target_link_libraries(logistic PRIVATE ${ANN})

target_compile_options(logistic
        PUBLIC -pipe
        $<$<CONFIG:Debug>:-O0 -ggdb3 -gsplit-dwarf>
        $<$<CONFIG:Release>:-O2 -march=native -flto -fno-fat-lto-objects> # release: 03 is implied, add back 02
        $<$<CONFIG:RelWithDebInfo>:-O2 -march=native -flto -fno-fat-lto-objects  -ggdb3 -gsplit-dwarf> # relwithdebinfo: 02 is implied
        -Wp,-D_FORTIFY_SOURCE=2  -Wp,-D_GLIBCXX_ASSERTIONS
        -fexceptions -fasynchronous-unwind-tables -fstack-protector-strong -fstack-clash-protection -fcf-protection
        -fPIE -Wl,-z,noexecstack,-z,relro,-z,defs,-pie,-z,now
        PRIVATE -Wall -Wextra -Wpedantic -Wformat=2 -Wswitch-default -Wswitch-enum -Wfloat-equal -Wno-conversion
        -pedantic-errors -Werror=format-security
        -fdiagnostics-color=always
    )
