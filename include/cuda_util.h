/*
 * util.h
 *
 *  Created on: Jun 18, 2021
 *      Author: mad
 */

#ifndef INCLUDE_MMX_CUDA_UTIL_H_
#define INCLUDE_MMX_CUDA_UTIL_H_

#include <string>
#include <stdexcept>

#include <cuda_runtime.h>
#include <cuda_runtime_util.h>


typedef long long int int64_cu;
typedef unsigned long long int uint64_cu;
//typedef uint64_t uint64_cu;


__device__ inline
uint32_t cuda_bswap_32(const uint32_t y) {
	return (y << 24) | ((y << 8) & 0xFF0000) | ((y >> 8) & 0xFF00) | (y >> 24);
}

__device__ inline
uint64_cu cuda_bswap_64(const uint64_cu y) {
	return (uint64_cu(cuda_bswap_32(y)) << 32) | cuda_bswap_32(y >> 32);
}

__device__ inline
uint32_t cuda_rotr_32(const uint32_t w, const uint32_t c) {
	return __funnelshift_r(w, w, c);
}

__device__ inline
uint32_t cuda_rotl_32(const uint32_t w, const uint32_t c) {
	return __funnelshift_l(w, w, c);
}

__device__ inline
uint64_t cuda_rotr_64(const uint64_t w, const uint32_t c) {
	return (w >> c) | (w << (64 - c));
}

__device__ inline
uint64_t cuda_rotl_64(const uint64_t w, const uint32_t c) {
	return (w << c) | (w >> (64 - c));
}

__global__ inline
void memset_u32(uint32_t* data, const uint32_t value, const uint64_cu count)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x < count) {
		data[x] = value;
	}
}

__global__ inline
void memset_u64(uint64_cu* data, const uint64_cu value, const uint64_cu count)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x < count) {
		data[x] = value;
	}
}

__global__ inline
void memset_u64_4(ulonglong4* data, const uint64_cu value, const uint64_cu count)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x < count) {
		data[x] = make_ulonglong4(value, value, value, value);
	}
}

__global__ inline
void memcpy_u32(uint32_t* dst, const uint32_t* src, const uint64_cu count)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x < count) {
		dst[x] = src[x];
	}
}

__global__ inline
void memcpy_u64(uint64_cu* dst, const uint64_cu* src, const uint64_cu count)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	if(x < count) {
		dst[x] = src[x];
	}
}

__global__ inline
void calc_offset_sum(uint32_t* offset_out, const uint32_t* count_in, const uint32_t width, const bool with_total)
{
	const int k = threadIdx.x;
	const int num_warps = blockDim.x / 32;

	__shared__ uint32_t local_offset[33];

	if(k == 0) {
		offset_out[0] = 0;
	}
	if(k < num_warps + 1) {
		local_offset[k] = 0;
	}
	__syncthreads();

	const uint32_t loop_count = (width + blockDim.x - 1) / blockDim.x;

	for(uint32_t y = 0; y < loop_count; ++y)
	{
		if(k < num_warps) {
			local_offset[k] = local_offset[num_warps];
		}
		const uint32_t x = y * blockDim.x + k;
		const uint32_t size_k = (x < width ? count_in[x] : 0);

		uint32_t warp_sum = size_k;
		for(int off = 16; off > 0; off /= 2) {
			warp_sum += __shfl_down_sync(0xFFFFFFFF, warp_sum, off);
		}
		warp_sum = __shfl_sync(0xFFFFFFFF, warp_sum, 0);

		__syncthreads();

		if(k % 32 >= k / 32 && k % 32 < num_warps) {
			atomicAdd(local_offset + 1 + k % 32, warp_sum);
		}
		uint32_t offset = size_k;
		for(int i = 0; i < 32; ++i) {
			const auto tmp = __shfl_sync(0xFFFFFFFF, size_k, i);
			if(i < k % 32) {
				offset += tmp;
			}
		}
		__syncthreads();

		offset += local_offset[k / 32];

		__syncthreads();

		if(x + 1 < width + (with_total ? 1 : 0)) {
			offset_out[x + 1] = offset;
		}
	}
}


template<typename T>
void cuda_memset(T* data, const T& value, size_t count, cudaStream_t stream)
{
	throw std::logic_error("not implemented");
}

template<> inline
void cuda_memset<uint32_t>(uint32_t* data, const uint32_t& value, size_t count, cudaStream_t stream)
{
	if(count >= (1 << 31)) {
		throw std::logic_error("count >= 2^31 - 1");
	}
	dim3 block(256, 1);
	dim3 grid((count + block.x - 1) / block.x, 1);
	memset_u32<<<grid, block, 0, stream>>>(data, value, count);
}

template<> inline
void cuda_memset<uint64_cu>(uint64_cu* data, const uint64_cu& value, size_t count, cudaStream_t stream)
{
	if(count >= (1 << 31)) {
		throw std::logic_error("count >= 2^31 - 1");
	}
	if(count % 4) {
		dim3 block(256, 1);
		dim3 grid((count + block.x - 1) / block.x, 1);
		memset_u64<<<grid, block, 0, stream>>>(data, value, count);
	} else {
		if(sizeof(ulonglong4) != sizeof(uint64_cu) * 4) {
			throw std::logic_error("sizeof(ulonglong4) != sizeof(uint64_cu) * 4");
		}
		count /= 4;
		dim3 block(256, 1);
		dim3 grid((count + block.x - 1) / block.x, 1);
		memset_u64_4<<<grid, block, 0, stream>>>((ulonglong4*)data, value, count);
	}
}

template<typename T>
void cuda_memcpy(T* dst, const T* src, size_t count, cudaStream_t stream)
{
	throw std::logic_error("not implemented");
}

template<> inline
void cuda_memcpy<uint32_t>(uint32_t* dst, const uint32_t* src, size_t count, cudaStream_t stream)
{
	if(count >= (1 << 31)) {
		throw std::logic_error("count >= 2^31 - 1");
	}
	dim3 block(256, 1);
	dim3 grid((count + block.x - 1) / block.x, 1);
	memcpy_u32<<<grid, block, 0, stream>>>(dst, src, count);
}

template<> inline
void cuda_memcpy<uint64_cu>(uint64_cu* dst, const uint64_cu* src, size_t count, cudaStream_t stream)
{
	if(count >= (1 << 31)) {
		throw std::logic_error("count >= 2^31 - 1");
	}
	dim3 block(256, 1);
	dim3 grid((count + block.x - 1) / block.x, 1);
	memcpy_u64<<<grid, block, 0, stream>>>(dst, src, count);
}




#endif /* INCLUDE_MMX_CUDA_UTIL_H_ */
