cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)

find_package(Git)
execute_process(
  COMMAND ${GIT_EXECUTABLE} log -1 --format=%h
  WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
  OUTPUT_VARIABLE GIT_COMMIT_HASH
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE "Release"
		CACHE STRING "Possible values are empty, Debug, Release, RelWithDebInfo, MinSizeRel, ..."
		FORCE
	)
endif()

project(cuda-plotter C CXX)

set(LIBBECH32_BUILD_TESTS "off" CACHE STRING "")
set(LIBBECH32_BUILD_EXAMPLES "off" CACHE STRING "")

add_subdirectory(vnx-base)
add_subdirectory(libbech32)

find_package(CUDA REQUIRED)
find_package(Threads REQUIRED)

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --disable-warnings")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_52,code=compute_52 -gencode arch=compute_75,code=compute_75 -gencode arch=compute_86,code=compute_86 -gencode arch=compute_89,code=compute_89")

if(MSVC)
	set(MSVC_DEBUG_OPTIONS /Od /FS)
	set(MSVC_RELEASE_OPTIONS /W1 /O2)

	add_compile_options( 
		"$<$<CONFIG:Debug>:${MSVC_DEBUG_OPTIONS}>"
		"$<$<CONFIG:RelWithDebInfo>:${MSVC_RELEASE_OPTIONS}>"
		"$<$<CONFIG:Release>:${MSVC_RELEASE_OPTIONS}>"
		"$<$<CONFIG:MinSizeRel>:${MSVC_RELEASE_OPTIONS}>"
    )
else()
	add_compile_options(-Wall -Wno-unused-function)
endif()

include_directories(
	lib
	include
	generated/include
	mad-direct-io/include
	uint256_t
	uint256_t/uint128_t
)

file(GLOB GENERATED_SOURCE_FILES generated/src/*.cpp)

add_library(mmx_iface STATIC
	${GENERATED_SOURCE_FILES}
)

add_library(uint256_t STATIC
	uint256_t/uint128_t/uint128_t.cpp
	uint256_t/uint256_t.cpp
)

target_compile_definitions(uint256_t PUBLIC _UINT256_T_STATIC _UINT128_T_STATIC)

target_link_libraries(mmx_iface vnx_base_static)

set(SRC_FILES
	src/Node_phase1.cu
	src/Node_phase2.cu
	src/Node.cu
	src/Bucket.cpp
	src/cuda_plot.cpp
)

set(ALL_LIBS mmx_iface bech32 uint256_t)

if(MSVC)
	set_target_properties(mmx_iface PROPERTIES LINK_OPTIONS "/NODEFAULTLIB:LIBCMT")
	
	list(APPEND ALL_LIBS ws2_32)
else()
	list(APPEND ALL_LIBS stdc++fs Threads::Threads)
endif()

cuda_add_executable(mmx_cuda_plot_k26 ${SRC_FILES})
cuda_add_executable(mmx_cuda_plot_k29 ${SRC_FILES})
cuda_add_executable(mmx_cuda_plot_k30 ${SRC_FILES})
cuda_add_executable(mmx_cuda_plot_k31 ${SRC_FILES})
cuda_add_executable(mmx_cuda_plot_k32 ${SRC_FILES})

target_link_libraries(mmx_cuda_plot_k26 ${ALL_LIBS})
target_link_libraries(mmx_cuda_plot_k29 ${ALL_LIBS})
target_link_libraries(mmx_cuda_plot_k30 ${ALL_LIBS})
target_link_libraries(mmx_cuda_plot_k31 ${ALL_LIBS})
target_link_libraries(mmx_cuda_plot_k32 ${ALL_LIBS})

target_compile_definitions(mmx_cuda_plot_k26 PUBLIC KSIZE=26 LOGBUCKETS=6)
target_compile_definitions(mmx_cuda_plot_k29 PUBLIC KSIZE=29 LOGBUCKETS=6)
target_compile_definitions(mmx_cuda_plot_k30 PUBLIC KSIZE=30 LOGBUCKETS=7)
target_compile_definitions(mmx_cuda_plot_k31 PUBLIC KSIZE=31 LOGBUCKETS=8)
target_compile_definitions(mmx_cuda_plot_k32 PUBLIC KSIZE=32 LOGBUCKETS=9)

target_compile_definitions(mmx_cuda_plot_k26 PRIVATE "GIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\"")
target_compile_definitions(mmx_cuda_plot_k29 PRIVATE "GIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\"")
target_compile_definitions(mmx_cuda_plot_k30 PRIVATE "GIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\"")
target_compile_definitions(mmx_cuda_plot_k31 PRIVATE "GIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\"")
target_compile_definitions(mmx_cuda_plot_k32 PRIVATE "GIT_COMMIT_HASH=\"${GIT_COMMIT_HASH}\"")

if(NOT MSVC)
# 	add_subdirectory(test)
endif()
