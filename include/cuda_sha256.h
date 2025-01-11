/*
 * cuda_sha256.h
 *
 *  Created on: Nov 4, 2023
 *      Author: mad
 */

#ifndef INCLUDE_MMX_CUDA_SHA256_H_
#define INCLUDE_MMX_CUDA_SHA256_H_

#include <cuda_util.h>

__device__
static const uint32_t SHA256_K[64] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
	0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
	0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
	0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
	0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
	0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
	0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2
};

__device__
static const uint32_t SHA256_INIT[8] = {
	0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19
};


#define SHA256_ZL25(n) ((cuda_rotl_32((n), 25) ^ cuda_rotl_32((n), 14) ^ ((n) >> 3U)))
#define SHA256_ZL15(n) ((cuda_rotl_32((n), 15) ^ cuda_rotl_32((n), 13) ^ ((n) >> 10U)))
#define SHA256_ZL26(n) ((cuda_rotl_32((n), 26) ^ cuda_rotl_32((n), 21) ^ cuda_rotl_32((n), 7)))
#define SHA256_ZL30(n) ((cuda_rotl_32((n), 30) ^ cuda_rotl_32((n), 19) ^ cuda_rotl_32((n), 10)))

//#define Ch(x, y, z) (z ^ (x & (y ^ z)))
//#define Ma(x, y, z) ((x & z) | (y & (x | z)))


__device__ inline
void cuda_sha256_chunk(const uint32_t* msg, uint32_t* state)
{
	uint32_t w[64];

	for(int i = 0; i < 16; ++i) {
		w[i] = cuda_bswap_32(msg[i]);
	}

//	for(int i = 16; i < 64; ++i)
//	{
//		const uint32_t s0 = ZR25(w[i-15]);
//		const uint32_t s1 = ZR15(w[i-2]);
//		w[i] = w[i-16] + s0 + w[i-7] + s1;
//	}

	uint32_t a = state[0];
	uint32_t b = state[1];
	uint32_t c = state[2];
	uint32_t d = state[3];
	uint32_t e = state[4];
	uint32_t f = state[5];
	uint32_t g = state[6];
	uint32_t h = state[7];

#pragma unroll
	for(int i = 0; i < 64; ++i)
	{
		if(i >= 16) {
			const uint32_t s0 = SHA256_ZL25(w[i-15]);
			const uint32_t s1 = SHA256_ZL15(w[i-2]);
			w[i] = w[i-16] + s0 + w[i-7] + s1;
		}

		const uint32_t S1 = SHA256_ZL26(e);
//		const uint32_t ch = (e & f) ^ ((~e) & g);
//		const uint32_t ch = Ch(e, f, g);
		const uint32_t ch = (g ^ (e & (f ^ g)));
		const uint32_t temp1 = h + S1 + ch + SHA256_K[i] + w[i];
		const uint32_t S0 = SHA256_ZL30(a);
//		const uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
//		const uint32_t maj = Ma(a, b, c);
		const uint32_t maj = ((a & c) | (b & (a | c)));
		const uint32_t temp2 = S0 + maj;

		h = g;
		g = f;
		f = e;
		e = d + temp1;
		d = c;
		c = b;
		b = a;
		a = temp1 + temp2;
	}

	state[0] += a;
	state[1] += b;
	state[2] += c;
	state[3] += d;
	state[4] += e;
	state[5] += f;
	state[6] += g;
	state[7] += h;
}

/*
 * msg needs 9 bytes buffer at the end and must be multiple of 16x 32-bit
 * msg needs to be zero initialized
 * maximum length = 2^29 bytes
 */
__device__ inline
void cuda_sha256(uint32_t* msg, const uint32_t length, uint32_t* hash)
{
	const uint32_t num_bits = length * 8;
	const uint32_t total_bytes = length + 9;
	const uint32_t num_chunks = (total_bytes + 63) / 64;

	msg[length / 4] |= (0x80 << ((length % 4) * 8));

	msg[num_chunks * 16 - 1] = cuda_bswap_32(num_bits);

	for(int i = 0; i < 8; ++i) {
		hash[i] = SHA256_INIT[i];
	}
	for(uint32_t i = 0; i < num_chunks; ++i) {
		cuda_sha256_chunk(msg + i * 16, hash);
	}
	for(int i = 0; i < 8; ++i) {
		hash[i] = cuda_bswap_32(hash[i]);
	}
}




#endif /* INCLUDE_MMX_CUDA_SHA256_H_ */
