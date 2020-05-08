cmake_minimum_required(VERSION 3.13)
project(info9 VERSION 1.0)

set(CMAKE_PREFIX_PATH, ${CONDA_PREFIX})


find_package(OpenMP REQUIRED)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)

set(HIGHFIVE_EXAMPLES FALSE)
set(HIGHFIVE_USE_BOOST FALSE)
set(HIGHFIVE_USE_EIGEN TRUE)
find_package(HighFive REQUIRED)

find_library(ANN NAMES ann)

# Define "info9-lib" library
add_subdirectory("../../src" "libinfo9")

add_executable(info9-pgm main.cpp)

target_link_libraries(info9-pgm PRIVATE info9-lib)
target_link_libraries(info9-pgm PRIVATE Eigen3::Eigen)
target_link_libraries(info9-pgm PRIVATE HighFive)

target_compile_options(info9-pgm
        PUBLIC -pipe
        $<$<CONFIG:Debug>:-O0 -ggdb3 -gsplit-dwarf>
        $<$<CONFIG:Release>:-O2 -march=native -flto -fno-fat-lto-objects> # release: 03 is implied, add back 02
        $<$<CONFIG:RelWithDebInfo>:-O2 -march=native -ggdb3 -flto -fno-fat-lto-objects> # relwithdebinfo: 02 is implied
        -Wp,-D_FORTIFY_SOURCE=2  -Wp,-D_GLIBCXX_ASSERTIONS
        -fexceptions -fasynchronous-unwind-tables -fstack-protector-strong -fstack-clash-protection -fcf-protection
        -fPIE -Wl,-z,noexecstack,-z,relro,-z,defs,-pie,-z,now
        PRIVATE -Wall -Wextra -Wpedantic -Wformat=2 -Wswitch-default -Wswitch-enum -Wfloat-equal -Wno-conversion
        -pedantic-errors -Werror=format-security
    )
