cmake_minimum_required(VERSION 3.23 FATAL_ERROR)
project(CUDA_stir LANGUAGES CXX CUDA)
set(CMAKE_CUDA_ARCHITECTURES "all")

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(BOOST_ROOT CACHE PATH "root of Boost")
find_package( Boost 1.36.0 REQUIRED )
find_package(STIR 6.0 REQUIRED CONFIG)
find_package(parallelproj 1.3.4 CONFIG)

set(SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src)
set(INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/include)

#file(GLOB_RECURSE SRC_FILES ${SRC_DIR}/*.cu ${SRC_DIR}/*.cxx)
#file(GLOB_RECURSE HEADER_FILES ${INCLUDE_DIR}/*.h)

add_executable(CPP_stir ${SRC_DIR}/main_cpp.cxx $<TARGET_OBJECTS:stir_registries>)
target_link_libraries(CPP_stir PUBLIC ${STIR_LIBRARIES})
target_include_directories(CPP_stir PUBLIC ${STIR_INCLUDE_DIRS})

add_executable(CUDA_stir 
    ${SRC_DIR}/main_cuda_class.cxx
    ${SRC_DIR}/CudaRelativeDifferencePriorClass.cu
    ${SRC_DIR}/CudaRelativeDifferencePriorKernels.cu
    ${INCLUDE_DIR}/CudaKernels.h
    ${INCLUDE_DIR}/CudaRelativeDifferencePriorClass.h
    $<TARGET_OBJECTS:stir_registries>)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(CUDA_stir PUBLIC CUDA::cudart)
target_link_libraries(CUDA_stir PUBLIC ${STIR_LIBRARIES})
target_include_directories(CUDA_stir PUBLIC ${STIR_INCLUDE_DIRS})
target_include_directories(CUDA_stir PUBLIC ${INCLUDE_DIR})
target_include_directories(CUDA_stir PUBLIC ${Boost_INCLUDE_DIRS})