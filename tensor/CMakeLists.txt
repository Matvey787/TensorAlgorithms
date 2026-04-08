cmake_minimum_required(VERSION 3.28)
project(Tensor_lib LANGUAGES CXX)



set(TENSOR_LIB tensor_gen)
add_library(${TENSOR_LIB})

target_compile_features(${TENSOR_LIB} PRIVATE cxx_std_23)

file(GLOB_RECURSE TENSOR_MODULES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cppm"
)

target_sources(${TENSOR_LIB}
    PUBLIC
        FILE_SET
            CXX_MODULES
        TYPE
            CXX_MODULES
        BASE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}/src
        FILES
            ${TENSOR_MODULES}
)

target_include_directories(${TENSOR_LIB}
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/nlohmann_json/include
)
