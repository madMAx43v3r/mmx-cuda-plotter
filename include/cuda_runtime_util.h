/*
 * cuda_runtime_util.h
 *
 *  Created on: Nov 23, 2023
 *      Author: mad
 */

#ifndef INCLUDE_MMX_CUDA_RUNTIME_UTIL_H_
#define INCLUDE_MMX_CUDA_RUNTIME_UTIL_H_

#include <string>
#include <stdexcept>

#include <cuda_runtime_api.h>


inline void cuda_check(const cudaError_t& code, const std::string& message = std::string()) {
	if(code != cudaSuccess) {
		throw std::runtime_error("CUDA error " + std::to_string(code) + ": " + message + std::string(cudaGetErrorString(code)));
	}
}

inline dim3 ceiled_grid_dim(uint32_t width, uint32_t height, dim3 block_dim) {
	return dim3((width + block_dim.x - 1) / block_dim.x, (height + block_dim.y - 1) / block_dim.y);
}

inline dim3 ceiled_grid_dim(uint32_t width, uint32_t height, uint32_t depth, dim3 block_dim) {
	return dim3((width + block_dim.x - 1) / block_dim.x,
				(height + block_dim.y - 1) / block_dim.y,
				(depth + block_dim.z - 1) / block_dim.z);
}



#endif /* INCLUDE_MMX_CUDA_RUNTIME_UTIL_H_ */
