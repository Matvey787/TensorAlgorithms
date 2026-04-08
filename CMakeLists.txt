cmake_minimum_required(VERSION 3.28)
project(LIRS_Cache LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23) 
set(CMAKE_CXX_STANDARD_REQUIRED ON)


add_subdirectory(tensor)


set(EXE_NAME winograd)

add_executable(${EXE_NAME} main.cpp)

target_link_libraries(${EXE_NAME} PRIVATE tensor_gen)

target_compile_features(${EXE_NAME} PRIVATE cxx_std_23)

