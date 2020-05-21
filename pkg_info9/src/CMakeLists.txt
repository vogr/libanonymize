cmake_minimum_required(VERSION 3.13)
project(info9-lib VERSION 1.0)


add_library(info9-lib STATIC
  "helpers.cpp" 
  "classification/ConfusionMatrix.cpp"
  "classification/ConfusionMatrixMulticlass.cpp"
  "classification/Dataset.cpp"
  "classification/KnnClassification.cpp"
  "classification/KnnClassificationBinary.cpp"
  "classification/KnnClassificationMulticlass.cpp"
  "classification/LogisticReg.cpp"
  "classification/LogisticRegMultinomial.cpp"
  "classification/RandomProjection.cpp"
)

set_target_properties(info9-lib PROPERTIES POSITION_INDEPENDENT_CODE ON)

set_target_properties(info9-lib PROPERTIES
    CXX_STANDARD 14
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)

target_include_directories(info9-lib PUBLIC ".")
target_include_directories(info9-lib PUBLIC "classification")


target_link_libraries(info9-lib PRIVATE Eigen3::Eigen)
target_link_libraries(info9-lib PRIVATE OpenMP::OpenMP_CXX)
target_link_libraries(info9-lib PRIVATE HighFive)
target_link_libraries(info9-lib PRIVATE ${ANN})

target_compile_options(info9-lib
    PUBLIC
        $<$<CONFIG:Debug>:-O0 -ggdb3 -gsplit-dwarf>
        $<$<CONFIG:Release>:-O2 -march=native -flto -fno-fat-lto-objects> # release: 03 is implied, add back 02
        $<$<CONFIG:RelWithDebInfo>:-O2 -march=native -ggdb3 -flto -fno-fat-lto-objects> # relwithdebinfo: 02 is implied
        -pipe
        -Wp,-D_FORTIFY_SOURCE=2 -Wp,-D_GLIBCXX_ASSERTIONS
        -fexceptions -fasynchronous-unwind-tables -fstack-protector-strong -fstack-clash-protection -fcf-protection
        -Wl,-z,noexecstack,-z,relro
    PRIVATE
        -Wall -Wextra -Wpedantic -Wformat=2 -Wswitch-default -Wswitch-enum -Wfloat-equal -Wno-conversion
        -pedantic-errors -Werror=format-security
    )
