cmake_minimum_required(VERSION 3.28)
project(Tensor_lib LANGUAGES CXX)

add_subdirectory(third_party/boost-json)

target_compile_definitions(boost_json PUBLIC BOOST_JSON_STANDALONE)


set(TENSOR_LIB tensor_lib)
add_library(${TENSOR_LIB})

target_compile_features(${TENSOR_LIB} PRIVATE cxx_std_23)

file(GLOB_RECURSE TENSOR_MODULES 
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cppm"
)

target_sources(${TENSOR_LIB}
    PUBLIC
        FILE_SET
            CXX_MODULES
        TYPE
            CXX_MODULES
        BASE_DIRS
            ${CMAKE_CURRENT_SOURCE_DIR}
        FILES
            ${TENSOR_MODULES}
)

target_link_libraries(${TENSOR_LIB} PRIVATE boost_json)