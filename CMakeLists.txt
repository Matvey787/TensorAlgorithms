cmake_minimum_required(VERSION 3.28)
project(LIRS_Cache LANGUAGES CXX)

include(cmake/debug_release.cmake)

add_subdirectory(tensor)

add_subdirectory(src/third_party/cxxopts)


set(EXE_NAME winograd)

add_executable(${EXE_NAME} src/main.cpp)

target_link_libraries(${EXE_NAME}
    PRIVATE
        tensor_gen
        cxxopts::cxxopts
)

file(GLOB_RECURSE EXE_MODULES
    "${CMAKE_SOURCE_DIR}/src/*.cppm"
)

target_sources(${EXE_NAME}
    PUBLIC
        FILE_SET
            CXX_MODULES
        TYPE
            CXX_MODULES
        FILES
            ${EXE_MODULES}
)

target_compile_features(${EXE_NAME} PRIVATE cxx_std_23)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_definitions(${EXE_NAME} PRIVATE DEBUG)
endif()






set(EXE_NAME2 winograd_compare)

add_executable(${EXE_NAME2} src/main.cpp)

target_link_libraries(${EXE_NAME2}
    PRIVATE
        tensor_gen
        cxxopts::cxxopts
)

file(GLOB_RECURSE EXE_MODULES
    "${CMAKE_SOURCE_DIR}/src/*.cppm"
)

target_sources(${EXE_NAME2}
    PUBLIC
        FILE_SET
            CXX_MODULES
        TYPE
            CXX_MODULES
        FILES
            ${EXE_MODULES}
)

target_compile_features(${EXE_NAME2} PRIVATE cxx_std_23)


target_compile_definitions(${EXE_NAME2} PRIVATE COMPARE)

