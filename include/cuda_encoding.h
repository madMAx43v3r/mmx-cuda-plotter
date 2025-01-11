/*
 * cuda_encoding.h
 *
 *  Created on: Jul 10, 2021
 *      Author: mad
 */

#ifndef INCLUDE_MMX_CUDA_ENCODING_H_
#define INCLUDE_MMX_CUDA_ENCODING_H_

#include <cuda_util.h>
#include <Buffer.h>


__device__ inline
uint64_t write_bits(uint64_cu* dst, uint64_t value, const uint64_t bit_offset, const uint32_t num_bits)
{
	if(num_bits < 64) {
		value &= ((uint64_t(1) << num_bits) - 1);
	}
	const uint32_t shift = bit_offset % 64;
	const uint32_t free_bits = 64 - shift;

	dst[bit_offset / 64]         |= (value << shift);

	if(free_bits < num_bits) {
		dst[bit_offset / 64 + 1] |= (value >> free_bits);
	}
	return bit_offset + num_bits;
}

__device__ inline
uint64_t atomic_write_bits(uint64_cu* dst, uint64_t value, const uint64_t bit_offset, const uint32_t num_bits)
{
	if(num_bits < 64) {
		value &= ((uint64_t(1) << num_bits) - 1);
	}
	const uint32_t shift = bit_offset % 64;
	const uint32_t free_bits = 64 - shift;

	atomicOr(dst + bit_offset / 64,         (value << shift));

	if(free_bits < num_bits) {
		atomicOr(dst + bit_offset / 64 + 1, (value >> free_bits));
	}
	return bit_offset + num_bits;
}

__device__ inline
uint64_t read_bits(const uint64_cu* src, const uint64_t bit_offset, const uint32_t num_bits)
{
	uint32_t count = 0;
	uint64_t offset = bit_offset;
	uint64_t result = 0;
	while(count < num_bits) {
		const uint32_t shift = offset % 64;
		const uint32_t bits = min(num_bits - count, 64 - shift);
		const uint64_t value = src[offset / 64] >> shift;
		result |= value << count;
		count += bits;
		offset += bits;
	}
	if(num_bits < 64) {
		result &= ((uint64_t(1) << num_bits) - 1);
	}
	return result;
}

__device__ inline
uint2 encode_symbol(const uint8_t sym)
{
	switch(sym) {
		case 0: return make_uint2(0, 2);
		case 1: return make_uint2(1, 2);
		case 2: return make_uint2(2, 2);
		case 3: return make_uint2(0b11 | (0 << 2), 4);
		case 4: return make_uint2(0b11 | (1 << 2), 4);
		case 5: return make_uint2(0b11 | (2 << 2), 4);
		case 6: return make_uint2(0b1111 | (0 << 4), 6);
		case 7: return make_uint2(0b1111 | (1 << 4), 6);
		case 8: return make_uint2(0b1111 | (2 << 4), 6);
	}
	const uint32_t index = sym / 3;
	const uint32_t mod = sym % 3;

	if(index > 15) {
		printf("encode_symbol(): out of range: %d\n", sym);
	}
	uint32_t out = uint32_t(-1) >> (32 - 2 * index);
	out |= mod << (2 * index);
	return make_uint2(out, 2 * index + 2);
}



#endif /* INCLUDE_MMX_CUDA_ENCODING_H_ */
