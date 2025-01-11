/*
 * Node_phase1.cu
 *
 *  Created on: Oct 9, 2021
 *      Author: mad
 */

#include <Node.h>

#include <config.h>
#include <cuda_util.h>
#include <cuda_sort.h>
#include <cuda_sha512.h>
#include <cuda_encoding.h>

#define MMXPOS_HASHROUND(a, b, c, d) \
	a = a + b;              \
	d = cuda_rotl_32(d ^ a, 16); \
	c = c + d;              \
	b = cuda_rotl_32(b ^ c, 12); \
	a = a + b;              \
	d = cuda_rotl_32(d ^ a, 8);  \
	c = c + d;              \
	b = cuda_rotl_32(b ^ c, 7);


__device__
static const uint32_t MEM_HASH_INIT[16] = {
	0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
	0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174
};

__global__
void gen_mem_array(uint4* mem_out, uint4* key_out, const uint32_t* id, const uint32_t mem_size, const uint32_t x_0)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t num_entries = gridDim.x * blockDim.x;

	__align__(8) uint32_t msg[32] = {};

	msg[0] = x_0 + x;

	for(int i = 0; i < 8; ++i) {
		msg[1 + i] = id[i];
	}
	__align__(8) uint32_t key[16] = {};

	cuda_sha512((uint64_t*)msg, 4 + 32, (uint64_t*)key);

	for(int i = 0; i < 4; ++i) {
		key_out[x * 4 + i] = make_uint4(key[i * 4 + 0], key[i * 4 + 1], key[i * 4 + 2], key[i * 4 + 3]);
	}

	uint32_t state[32];
	for(int i = 0; i < 16; ++i) {
		state[i] = key[i];
	}
	for(int i = 0; i < 16; ++i) {
		state[16 + i] = MEM_HASH_INIT[i];
	}

	uint32_t b = 0;
	uint32_t c = 0;

	for(uint32_t i = 0; i < mem_size / 32; ++i)
	{
		for(int j = 0; j < 4; ++j) {
#pragma unroll
			for(int k = 0; k < 16; ++k) {
				MMXPOS_HASHROUND(state[k], b, c, state[16 + k]);
			}
		}

#pragma unroll
		for(int k = 0; k < 8; ++k) {
			mem_out[(uint64_t(i) * num_entries + x) * 8 + k] =
					make_uint4(state[k * 4 + 0], state[k * 4 + 1], state[k * 4 + 2], state[k * 4 + 3]);
		}
	}
}

__global__
void calc_mem_hash(uint32_t* mem, uint32_t* hash, const int num_iter)
{
	const uint32_t x = threadIdx.x;
	const uint32_t k = threadIdx.y;
	const uint32_t y = (blockIdx.z * gridDim.y + blockIdx.y) * blockDim.y + threadIdx.y;
	const uint32_t num_entries = (gridDim.z * blockDim.z) * (gridDim.y * blockDim.y);

	static constexpr int N = 32;

	__shared__ uint32_t lmem[4][N*N];

	for(int i = 0; i < N; ++i) {
		lmem[k][i * N + x] = mem[(uint64_t(i) * num_entries + y) * N + x];
	}
	__syncwarp();

	uint32_t state = lmem[k][(N - 1) * N + x];

	__syncwarp();

	for(int iter = 0; iter < num_iter; ++iter)
	{
		uint32_t sum = cuda_rotl_32(state, x % 32);

		for(int offset = 16; offset > 0; offset /= 2) {
			sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);
		}
		uint32_t dir = 0;
		if(x == 0) {
			dir = sum + (sum << 11) + (sum << 22);
		}
		sum = __shfl_sync(0xFFFFFFFF, sum, 0);
		dir = __shfl_sync(0xFFFFFFFF, dir, 0);

		const uint32_t bits = (dir >> 22) % 32u;
		const uint32_t offset = (dir >> 27);

		state += cuda_rotl_32(lmem[k][offset * N + (iter + x) % N], bits) ^ sum;

		__syncwarp();

		atomicXor(&lmem[k][offset * N + x], state);

		__syncwarp();
	}

	hash[y * N + x] = state;
}

__global__
void scatter_t1(uint32_t* X_out, uint32_t* C_out, uint32_t* bucket_size,
				const uint4* H_in, const uint4* key_in, const uint32_t X_0, const uint32_t max_bucket_size)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

	__align__(8) uint32_t msg[64] = {};

	for(int i = 0; i < 4; ++i) {
		const auto tmp = key_in[x * 4 + i];
		msg[i * 4 + 0] = tmp.x;
		msg[i * 4 + 1] = tmp.y;
		msg[i * 4 + 2] = tmp.z;
		msg[i * 4 + 3] = tmp.w;
	}
	for(int i = 0; i < 8; ++i) {
		const auto tmp = H_in[x * 8 + i];
		msg[16 + i * 4 + 0] = tmp.x;
		msg[16 + i * 4 + 1] = tmp.y;
		msg[16 + i * 4 + 2] = tmp.z;
		msg[16 + i * 4 + 3] = tmp.w;
	}
	__align__(8) uint32_t hash[16] = {};

	cuda_sha512((uint64_t*)msg, 64 + 128, (uint64_t*)hash);

	uint32_t Y_i = 0;
	for(int i = 0; i < N_META; ++i) {
		Y_i = Y_i ^ hash[i];
	}
	Y_i &= KMASK;

	const uint32_t index = Y_i >> (KSIZE - LOGBUCKETS);
	if((index >> LOGBUCKETS) == 0) {
		const uint32_t pos = atomicAdd(bucket_size + index, 1);
		if(pos < max_bucket_size) {
			const uint32_t j = index * max_bucket_size + pos;
			for(int i = 0; i < N_META; ++i) {
				C_out[j * N_META + i] = hash[i] & KMASK;
			}
			X_out[j] = X_0 + x;
		}
	}
}

__global__
void scatter_2(	uint64_cu* PY_out, uint32_t* bucket_size_out, const uint32_t* Y_in, const uint32_t* C_in,
				const uint32_t bucket_size_in, const uint32_t max_bucket_size_2)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

	if(x >= bucket_size_in) {
		return;
	}
	uint32_t Y_i = 0;

	if(Y_in) {
		Y_i = Y_in[x];
	} else {
		for(int i = 0; i < N_META; ++i) {
			Y_i = Y_i ^ C_in[x * N_META + i];
		}
		Y_i &= KMASK;
	}

	const uint32_t index = (Y_i >> (KSIZE - LOGBUCKETS - LOGBUCKETS2)) & ((uint32_t(1) << LOGBUCKETS2) - 1);
	const uint32_t pos = atomicAdd(bucket_size_out + index, 1);
	if(pos < max_bucket_size_2) {
		const uint32_t j = index * max_bucket_size_2 + pos;
		PY_out[j] = (uint64_t(Y_i) << (64 - KSIZE)) | x;
	}
}

__global__
void hybrid_sort_y(uint64_cu* data, const uint32_t* bucket_size, const uint32_t max_bucket_size)
{
	const uint32_t x = threadIdx.x;
	const uint32_t y = blockIdx.z * gridDim.y + blockIdx.y;

	static constexpr int LOG_THREADS = HYBRID_SORT_LOG_THREADS;
	static constexpr int NUM_THREADS = 1 << LOG_THREADS;
	static constexpr int AVG_BUCKET_SIZE = 1 << (KSIZE - LOGBUCKETS - LOGBUCKETS2);
	static constexpr int MAX_LOCAL_SIZE = (2 * AVG_BUCKET_SIZE) / NUM_THREADS + 24;

	if(MAX_LOCAL_SIZE > NUM_THREADS) {
		printf("hybrid_sort_y(): MAX_LOCAL_SIZE > NUM_THREADS\n");
	}
	const uint32_t size = min(bucket_size[y], max_bucket_size);

	__shared__ uint64_cu buffer[MAX_LOCAL_SIZE][NUM_THREADS + 1];
	__shared__ uint32_t  count[NUM_THREADS];

	count[x] = 0;

	__syncthreads();

	for(uint32_t i = x; i < size; i += NUM_THREADS)
	{
		const uint64_t PY = data[y * max_bucket_size + i];
		const uint32_t index = (PY >> (64 - LOGBUCKETS - LOGBUCKETS2 - LOG_THREADS)) & (NUM_THREADS - 1);
		if(index < NUM_THREADS) {
			const auto j = atomicAdd(count + index, 1);
			if(j < MAX_LOCAL_SIZE) {
				buffer[j][index] = PY;
			}
		} else {
			printf("hybrid_sort_y(): index >= NUM_THREADS: %d\n", index);
		}
	}
	__syncthreads();

	insertion_sort<uint64_cu>(x, min(count[x], MAX_LOCAL_SIZE), NUM_THREADS + 1, &buffer[0][0]);

	__syncthreads();

	uint32_t off = 0;
	for(int i = 0; i < NUM_THREADS; ++i) {
		const auto len = min(count[i], MAX_LOCAL_SIZE);
		if(x < len) {
			data[y * max_bucket_size + off + x] = buffer[x][i];
		}
		off += len;
	}
	if(x == 0 && off < size) {
		printf("hybrid_sort_y(): bucket overflow: %d entries lost\n", size - off);
	}
}

__global__
void write_x2(	uint64_cu* X_out, const uint64_cu* X_in, const uint64_cu* PY_in,
				const uint32_t* bucket_size, const uint32_t* bucket_offset,
				const uint32_t max_bucket_size, const int X2SIZE, const int XBITS)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.z * gridDim.y + blockIdx.y;

	if(x >= min(bucket_size[y], max_bucket_size)) {
		return;
	}
	const uint32_t offset = bucket_offset[y] + x;
	const uint32_t P_i = PY_in[y * max_bucket_size + x];

	const uint32_t X_1 = read_bits(X_in, uint64_t(P_i) * X2SIZE, XBITS);
	const uint32_t X_2 = read_bits(X_in, uint64_t(P_i) * X2SIZE + XBITS, XBITS);

	atomic_write_bits(X_out, X_1, uint64_t(offset) * X2SIZE, XBITS);
	atomic_write_bits(X_out, X_2, uint64_t(offset) * X2SIZE + XBITS, XBITS);
}

__global__
void write_pd(	uint64_cu* PD_out, const uint64_cu* PD_in, const uint64_cu* PY_in,
				const uint32_t* bucket_size, const uint32_t* bucket_offset,
				const uint32_t max_bucket_size)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.z * gridDim.y + blockIdx.y;

	if(x >= min(bucket_size[y], max_bucket_size)) {
		return;
	}
	const uint32_t offset = bucket_offset[y] + x;
	const uint32_t P_i = PY_in[y * max_bucket_size + x];

	const uint64_t PD = read_bits(PD_in, uint64_t(P_i) * PDSIZE, PDSIZE);

	atomic_write_bits(PD_out, PD, uint64_t(offset) * PDSIZE, PDSIZE);
}

__global__
void write_y(	uint32_t* Y_out, const uint64_cu* PY_in,
				const uint32_t* bucket_size, const uint32_t* bucket_offset,
				const uint32_t max_bucket_size)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.z * gridDim.y + blockIdx.y;

	if(x >= min(bucket_size[y], max_bucket_size)) {
		return;
	}
	const uint32_t offset = bucket_offset[y] + x;
	Y_out[offset] = PY_in[y * max_bucket_size + x] >> (64 - KSIZE);
}

__global__
void write_meta(uint32_t* M_out, const uint32_t* M_in, const uint64_cu* PY_in,
				const uint32_t* bucket_size, const uint32_t* bucket_offset,
				const uint32_t max_bucket_size)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.z * gridDim.y + blockIdx.y;

	if(x >= min(bucket_size[y], max_bucket_size)) {
		return;
	}
	const uint32_t offset = bucket_offset[y] + x;
	const uint32_t P_i = PY_in[y * max_bucket_size + x];

	for(int i = 0; i < N_META_OUT; ++i) {
		M_out[offset * N_META_OUT + i] = M_in[P_i * N_META_OUT + i];
	}
}

__global__
void match_p1(	uint2* LR_out, uint32_t* PD_out, uint32_t* num_matches,
				const uint64_cu* PY_in, const uint32_t* bucket_size, const uint32_t* bucket_offset,
				const uint32_t num_buckets, const uint32_t max_bucket_size, const uint32_t max_total_matches)
{
	const uint32_t k = threadIdx.x;
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;
	const uint32_t y = blockIdx.z * gridDim.y + blockIdx.y;

	static constexpr int MAX_MATCHES = 512;

	__shared__ uint2    LR_tmp[MAX_MATCHES];
	__shared__ uint32_t PD_tmp[MAX_MATCHES];
	__shared__ uint32_t count;
	__shared__ uint32_t global_offset;

	const uint32_t size = min(bucket_size[y], max_bucket_size);

	if(k == 0) {
		count = 0;
	}
	if(__syncthreads_and(x >= size)) {
		return;
	}
	const uint32_t P_x = bucket_offset[y] + x;
	const uint32_t next_size = (y + 1 < num_buckets ? bucket_size[y + 1] : 0);

	if(x < size)
	{
		const uint64_t PY_L = __ldg(PY_in + y * max_bucket_size + x);
		const uint32_t YL = PY_L >> (64 - KSIZE);
		const uint32_t PL = PY_L;

		for(uint32_t i = x + 1; true; ++i)
		{
			const bool is_next = (i >= size);
			const uint32_t j = (is_next ? i - size : i);

			if(is_next && j >= next_size) {
				break;
			}
			const uint64_t PY_R = __ldg(PY_in + (y + is_next) * max_bucket_size + j);
			const uint32_t YR = PY_R >> (64 - KSIZE);
			const uint32_t PR = PY_R;

			if(YR == YL + 1) {
				const auto pos = atomicInc(&count, 0xFFFFFFFF);
				if(pos < MAX_MATCHES) {
					LR_tmp[pos] = make_uint2(PL, PR);
					if(PD_out) {
						PD_tmp[pos] = (P_x << DSIZE_) | uint32_t(i - x - 1);
					}
				}
			} else if(YR > YL) {
				break;
			}
		}
	}
	__syncthreads();

	if(k == 0) {
		count = min(count, MAX_MATCHES);
		global_offset = atomicAdd(num_matches, count);
	}
	__syncthreads();

	const uint32_t offset = global_offset;

	for(uint32_t i = k; i < count; i += blockDim.x)
	{
		const uint32_t pos = offset + i;
		if(pos < max_total_matches) {
			LR_out[pos] = LR_tmp[i];
			if(PD_out) {
				PD_out[pos] = PD_tmp[i];
			}
		}
	}
}

__global__
void eval_p1_tx(uint32_t* Y_out, uint32_t* C_out, uint64_cu* PD_out, uint32_t* bucket_size,
				const uint32_t* C_in, const uint32_t* PD_in, const uint32_t* X_in,
				const uint2* LR_in, const uint32_t* num_found, const uint64_t PD_0,
				const uint32_t max_bucket_size, const int X2SIZE, const int XBITS, const int table)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

	if(x >= num_found[0]) {
		return;
	}
	const uint2 LR_i = LR_in[x];
	const uint32_t P_1 = LR_i.x;
	const uint32_t P_2 = LR_i.y;

	__align__(8) uint32_t hash[16] = {};
	__align__(8) uint32_t block[64] = {};

	for(int i = 0; i < N_META; ++i) {
		block[i] = 			C_in[P_1 * N_META + i];
		block[N_META + i] = C_in[P_2 * N_META + i];
	}
	cuda_sha512((uint64_t*)block, 2 * META_BYTES, (uint64_t*)hash);

	uint32_t Y_new = 0;
	for(int i = 0; i < N_META; ++i) {
		Y_new = Y_new ^ hash[i];
	}
	Y_new &= KMASK;

	const uint32_t index = Y_new >> (KSIZE - LOGBUCKETS);
	if((index >> LOGBUCKETS) == 0) {
		const uint32_t pos = atomicAdd(bucket_size + index, 1);
		if(pos < max_bucket_size) {
			const uint64_t j = index * max_bucket_size + pos;
			if(Y_out) {
				Y_out[j] = Y_new;
			}
			if(C_out) {
				if(table < N_TABLE) {
					for(int i = 0; i < N_META; ++i) {
						C_out[j * N_META + i] = hash[i] & KMASK;
					}
				} else {
					for(int i = 0; i < N_META_OUT; ++i) {
						C_out[j * N_META_OUT + i] = hash[i] & KMASK;
					}
				}
			}
			if(PD_in) {
				atomic_write_bits(PD_out, PD_0 + PD_in[x], j * PDSIZE, PDSIZE);
			}
			if(X_in) {
				atomic_write_bits(PD_out, X_in[P_1] >> (KSIZE - XBITS), j * X2SIZE, XBITS);
				atomic_write_bits(PD_out, X_in[P_2] >> (KSIZE - XBITS), j * X2SIZE + XBITS, XBITS);
			}
		}
	}
}



void Node::phase1()
{
	if(max_bucket_size_1 % 256) {
		throw std::logic_error("max_bucket_size_1 % 256 != 0");
	}
	if(num_buckets_2 % 256) {
		throw std::logic_error("num_buckets_2 % 256 != 0");
	}
	const auto p1_begin = get_time_millis();

	if(!is_flushed) {
		Bucket::block_alloc();
	}
	const uint32_t mem_size = MEM_HASH_N * MEM_HASH_N;
	const uint64_t hash_num_buckets_1 = (uint64_t(1) << KSIZE) / num_buckets_1;
	const uint64_t hash_block_size = uint64_t(1) << 16;
	const uint32_t hash_block_count = cdiv(hash_num_buckets_1, hash_block_size);

	Buffer<uint32_t>*  mem_buf = new Buffer<uint32_t>[NSTREAMS];
	Buffer<uint32_t>*  key_buf = new Buffer<uint32_t>[NSTREAMS];
	Buffer<uint32_t>*  hash_buf = new Buffer<uint32_t>[NSTREAMS];

	Buffer<uint32_t>*  Y_buf = new Buffer<uint32_t>[NSTREAMS];
	Buffer<uint32_t>*  C_buf = new Buffer<uint32_t>[NSTREAMS];
	Buffer<uint32_t>*  Y_out = new Buffer<uint32_t>[NSTREAMS];
	Buffer<uint32_t>*  Y_in = new Buffer<uint32_t>[NSTREAMS];
	Buffer<uint32_t>*  C_in = new Buffer<uint32_t>[NSTREAMS];
//	Buffer<uint32_t>*  Y_tmp = new Buffer<uint32_t>[NSTREAMS];
	Buffer<uint32_t>*  C_out = new Buffer<uint32_t>[NSTREAMS];
	Buffer<uint64_cu>* PD_in = new Buffer<uint64_cu>[NSTREAMS];
	Buffer<uint64_cu>* PD_out = new Buffer<uint64_cu>[NSTREAMS];
	Buffer<uint64_cu>* PD_buf = new Buffer<uint64_cu>[NSTREAMS];

	for(int i = 0; i < NSTREAMS; ++i) {
		cudaSetDevice(device + i % num_devices);

		mem_buf[i].alloc(	hash_block_size * mem_size, MEM_TYPE_DEVICE);
		key_buf[i].alloc(	hash_block_size * 16, MEM_TYPE_DEVICE);
		hash_buf[i].alloc(	hash_block_size * 32, MEM_TYPE_DEVICE);

		Y_out[i].alloc(		std::max(max_entries_tmp, max_bucket_size_1), MEM_TYPE_DEVICE);
		C_out[i].alloc(		std::max(max_entries_tmp, max_bucket_size_1) * N_META, MEM_TYPE_DEVICE);
		Y_in[i].alloc(		max_bucket_size_1, MEM_TYPE_DEVICE);
		C_in[i].alloc(		max_bucket_size_1 * N_META, MEM_TYPE_DEVICE);
//		Y_tmp[i].alloc(		max_bucket_size_1, MEM_TYPE_DEVICE);
		Y_buf[i].alloc(		std::max(max_entries_tmp, max_bucket_size_1), MEM_TYPE_PINNED);
		C_buf[i].alloc(		std::max(max_entries_tmp, max_bucket_size_1) * N_META, MEM_TYPE_PINNED);
		PD_in[i].alloc(		cdiv(max_bucket_size_1 * std::max({PDBYTES, X2BYTES, KBYTES}), 8), MEM_TYPE_DEVICE);
		PD_out[i].alloc(	cdiv(max_entries_tmp * std::max({PDBYTES, X2BYTES, KBYTES}), 8), MEM_TYPE_DEVICE);
		PD_buf[i].alloc(	cdiv(max_entries_tmp * std::max({PDBYTES, X2BYTES, KBYTES}), 8), MEM_TYPE_PINNED);
	}

	std::cout << "[P1] Setup took " << (get_time_millis() - p1_begin) / 1e3 << " sec" << std::endl;

	const auto t1_begin = get_time_millis();
	{
		table_result_t result;

		C_buckets[1] = alloc_buckets(file_store2.get());
		PD_buckets[1] = alloc_buckets(file_store.get());

		bucket_size_recv.memset_cpu(0);

		::memcpy(plot_enc_key.data(), params.id.data(), params.id.size());
		cuda_check(cudaMemcpy(plot_enc_key_in.data(), plot_enc_key.data(), plot_enc_key.num_bytes(), cudaMemcpyHostToDevice));

		reset_barriers();

		// start download threads
		for(int k = 0; k < NSTREAMS; ++k) {
			download_thread[k] = std::thread([this, k, C_buf, Y_buf, PD_buf, &result]() {
				try {
					for(uint32_t y = k; y < num_buckets_1; y += NSTREAMS)
					{
						wait_barrier(stream_barrier[y]);
						cuda_check(cudaEventSynchronize(download_event[y]));

						const auto* bucket_size_in = bucket_size_buf[k].data();

						const auto bucket_offset = update_chunk(bucket_size_recv.data(), bucket_size_in, max_bucket_size_tmp);

						sync_chunk_single(C_buckets[1],   C_buf[k].data(), bucket_offset.data(), bucket_size_in, max_bucket_size_1, max_bucket_size_tmp, META_BYTES);
						sync_chunk_single(PD_buckets[1], PD_buf[k].data(), bucket_offset.data(), bucket_size_in, max_bucket_size_1, max_bucket_size_tmp, KBYTES);

						signal_barrier(download_barrier[y]);

						for(size_t i = 0; i < num_buckets_1; ++i) {
							if(bucket_size_in[i] > max_bucket_size_tmp) {
								std::cerr << "WARN: max_bucket_size_tmp overflow" << std::endl;
							}
							result.max_bucket_size_tmp = std::max(result.max_bucket_size_tmp, bucket_size_in[i]);
						}
					}
				} catch(const std::exception& ex) {
					std::lock_guard<std::mutex> lock(cout_mutex);
					std::cerr << "T1 download thread failed with: " << ex.what() << std::endl;
					throw;
				}
			});
		}

		for(uint32_t y = 0; y < num_buckets_1; ++y)
		{
			const auto k = y % NSTREAMS;
			const auto stream = streams[k];

			cudaSetDevice(device + k % num_devices);

			cuda_memset<uint32_t>(bucket_size_out[k], 0, stream);

			for(uint32_t i = 0; i < hash_block_count; ++i)
			{
				const uint32_t X_0 = y * hash_num_buckets_1 + i * hash_block_size;
				{
					dim3 block(256, 1);
					dim3 grid(hash_block_size / block.x, 1);
					gen_mem_array<<<grid, block, 0, stream>>>(
							(uint4*)mem_buf[k].data(),
							(uint4*)key_buf[k].data(),
							plot_enc_key_in.data(),
							mem_size, X_0);
				}
				{
					dim3 block(MEM_HASH_N, 4);
					dim3 grid(1, hash_block_size / block.y / 16, 16);
					calc_mem_hash<<<grid, block, 0, stream>>>(
							mem_buf[k].data(),
							hash_buf[k].data(),
							MEM_HASH_ITER);
				}
				{
					dim3 block(256, 1);
					dim3 grid(hash_block_size / block.x, 1);
					scatter_t1<<<grid, block, 0, stream>>>(
							(uint32_t*)PD_out[k].data(),
							C_out[k].data(),
							bucket_size_out[k].data(),
							(const uint4*)hash_buf[k].data(),
							(const uint4*)key_buf[k].data(),
							X_0,
							max_bucket_size_tmp);
				}
			}

			if(y >= NSTREAMS) {
				wait_barrier(download_barrier[y - NSTREAMS]);
			}
			download_buffer(C_buf[k], C_out[k], max_entries_tmp * META_BYTES, stream);
			download_buffer(PD_buf[k], PD_out[k], max_entries_tmp * KBYTES, stream);
			download_buffer(bucket_size_buf[k], bucket_size_out[k], stream);

			cudaEventRecord(download_event[y], stream);

			signal_barrier(stream_barrier[y]);
		}

		for(int k = 0; k < NSTREAMS; ++k) {
			download_thread[k].join();
		}
		flush_buckets(C_buckets[1]);
		flush_buckets(PD_buckets[1]);

		const double elapsed = (get_time_millis() - t1_begin) / 1e3;

		for(size_t i = 0; i < num_buckets_1; ++i) {
			bucket_size_1[1][i] = std::min<uint32_t>(bucket_size_recv[i], max_bucket_size_1);
		}
		for(int i = 0; i < num_buckets_1; ++i) {
			result.num_entries += bucket_size_1[1][i];
			result.max_bucket_size = std::max(result.max_bucket_size, bucket_size_1[1][i]);
		}
		result.num_bytes_upload = g_upload_bytes;
		result.num_bytes_download = g_download_bytes;

		std::cout << "[P1] Table 1 took " << elapsed << " sec, "
				<< result.num_entries << " entries, " << result.max_bucket_size << " max, " << result.max_bucket_size_tmp << " tmp, "
				<< g_upload_bytes / elapsed / pow(1024, 3) << " GB/s up, "
				<< g_download_bytes / elapsed / pow(1024, 3) << " GB/s down" << std::endl;
		g_upload_bytes = 0;
		g_download_bytes = 0;
	}

	delete [] mem_buf;
	delete [] key_buf;
	delete [] hash_buf;

	Buffer<uint32_t>*  num_matches = new Buffer<uint32_t>[NSTREAMS];
	Buffer<uint32_t>*  bucket_size_2 = new Buffer<uint32_t>[NSTREAMS];
	Buffer<uint32_t>*  bucket_offset_2 = new Buffer<uint32_t>[NSTREAMS];
	Buffer<uint64_cu>* PY_tmp = new Buffer<uint64_cu>[NSTREAMS];
	Buffer<uint2>*     LR_tmp = new Buffer<uint2>[NSTREAMS];
	Buffer<uint32_t>*  PD_tmp = new Buffer<uint32_t>[NSTREAMS];
	Buffer<uint64_cu>* PD1_out = new Buffer<uint64_cu>[NSTREAMS];
	Buffer<uint64_cu>* PD1_buf = new Buffer<uint64_cu>[NSTREAMS];

	for(int i = 0; i < NSTREAMS; ++i) {
		cudaSetDevice(device + i % num_devices);

		num_matches[i].alloc(		1, MEM_TYPE_DEVICE);
		bucket_size_2[i].alloc(		num_buckets_2, MEM_TYPE_DEVICE);
		bucket_offset_2[i].alloc(	num_buckets_2, MEM_TYPE_DEVICE);
		PY_tmp[i].alloc(			num_buckets_2 * max_bucket_size_2, MEM_TYPE_DEVICE);
		LR_tmp[i].alloc(			max_bucket_size_1, MEM_TYPE_DEVICE);
		PD_tmp[i].alloc(			max_bucket_size_1, MEM_TYPE_DEVICE);
		PD1_out[i].alloc(			cdiv(max_entries_tmp * std::max(PDBYTES, X2BYTES), 8), MEM_TYPE_DEVICE);
		PD1_buf[i].alloc(			cdiv(max_entries_tmp * std::max(PDBYTES, X2BYTES), 8), MEM_TYPE_PINNED);
	}

	if(file_store) {
		file_store->flush();
	}
	if(file_store2) {
		file_store2->flush();
	}

	for(int t = 2; t <= N_TABLE + 1; ++t)
	{
		table_result_t result;
		const int src = (t+1) % 2;
		const int dst = t % 2;

		const auto tx_begin = get_time_millis();

		if(t >= N_TABLE) {
			Y_buckets[dst] = alloc_buckets(file_store2.get());
		}
		if(t <= N_TABLE) {
			PD_buckets[dst] = alloc_buckets(file_store.get());
		}
		if(t > final_table) {
			PD1_buckets[t-1] = alloc_buckets(file_store.get(), 0, false);
		}
		C_buckets[dst] = alloc_buckets(file_store2.get());

		bucket_size_recv.memset_cpu(0);

		reset_barriers();

		// start upload threads
		for(int k = 0; k < NSTREAMS; ++k) {
			upload_thread[k] = std::thread([this, t, k, src, Y_in, C_in, PD_in]() {
				try {
					for(uint32_t y = k; y < num_buckets_1; y += NSTREAMS)
					{
						if(y >= NSTREAMS) {
							wait_barrier(stream_barrier[y - NSTREAMS]);
						}
						const auto stream = streams[k];

						if(t > N_TABLE) {
							g_upload_bytes += Y_buckets[src][y]->upload(Y_in[k], stream);
						}
						g_upload_bytes += C_buckets[src][y]->upload(C_in[k], stream);
						g_upload_bytes += PD_buckets[src][y]->upload(PD_in[k], stream);
						cuda_check(cudaEventRecord(upload_event[y], stream));

						signal_barrier(upload_barrier[y]);

						cuda_check(cudaEventSynchronize(upload_event[y]));

						if(t > N_TABLE) {
							Y_buckets[src][y]->free();
						}
						C_buckets[src][y]->free();
						PD_buckets[src][y]->free();
					}
				} catch(const std::exception& ex) {
					std::lock_guard<std::mutex> lock(cout_mutex);
					std::cerr << "P1 upload thread failed with: " << ex.what() << std::endl;
					throw;
				}
			});
		}

		// start download threads
		for(int k = 0; k < NSTREAMS; ++k) {
			download_thread[k] = std::thread([this, t, k, dst, Y_buf, C_buf, PD_buf, PD1_buf, &result]() {
				try {
					for(uint32_t y = k; y < num_buckets_1; y += NSTREAMS)
					{
						wait_barrier(stream_barrier[y]);
						cuda_check(cudaEventSynchronize(download_event[y]));

						const uint64_t bucket_size_t1y = bucket_size_1[t-1][y];

						if(t <= N_TABLE) {
							const auto* bucket_size_in = bucket_size_buf[k].data();

							const auto bucket_offset = update_chunk(bucket_size_recv.data(), bucket_size_in, max_bucket_size_tmp);

							if(t == N_TABLE) {
								sync_chunk_single(Y_buckets[dst], Y_buf[k].data(), bucket_offset.data(), bucket_size_in, max_bucket_size_1, max_bucket_size_tmp, YBYTES);
							}
							if(t < N_TABLE || is_hdd_plot) {
								sync_chunk_single(C_buckets[dst], C_buf[k].data(), bucket_offset.data(), bucket_size_in, max_bucket_size_1, max_bucket_size_tmp, (t < N_TABLE ? META_BYTES : META_BYTES_OUT));
							}
							sync_chunk_single(PD_buckets[dst], PD_buf[k].data(), bucket_offset.data(), bucket_size_in, max_bucket_size_1, max_bucket_size_tmp, (t == 2 ? X2BYTES : PDBYTES));

							for(size_t i = 0; i < num_buckets_1; ++i) {
								if(bucket_size_in[i] > max_bucket_size_tmp) {
									std::cerr << "WARN: max_bucket_size_tmp overflow" << std::endl;
								}
								result.max_bucket_size_tmp = std::max(result.max_bucket_size_tmp, bucket_size_in[i]);
							}
						} else {
							if(is_hdd_plot) {
								C_buckets[dst][y]->copy(C_buf[k].data(), 0, bucket_size_t1y * META_BYTES_OUT);
							}
							Y_buckets[dst][y]->copy(Y_buf[k].data(), 0, bucket_size_t1y * YBYTES);
						}
						if(t > final_table) {
							PD1_buckets[t-1][y]->copy(PD1_buf[k].data(), 0, bucket_size_t1y * (t == 3 ? X2BYTES : PDBYTES));
						}
						signal_barrier(download_barrier[y]);
					}
				} catch(const std::exception& ex) {
					std::lock_guard<std::mutex> lock(cout_mutex);
					std::cerr << "P1 download thread failed with: " << ex.what() << std::endl;
					throw;
				}
			});
		}

		for(uint32_t y = 0; y < num_buckets_1; ++y)
		{
			const auto k = y % NSTREAMS;
			const auto stream = streams[k];

			const int key_size_2 = KSIZE - LOGBUCKETS - LOGBUCKETS2;

			cudaSetDevice(device + k % num_devices);

			wait_barrier(upload_barrier[y]);

			cuda_memset<uint32_t>(bucket_size_2[k], 0, stream);
			{
				dim3 block(256, 1);
				dim3 grid(max_bucket_size_1 / block.x, 1);
				scatter_2<<<grid, block, 0, stream>>>(
						PY_tmp[k].data(),
						bucket_size_2[k].data(),
						t > N_TABLE ? Y_in[k].data() : nullptr,
						C_in[k].data(),
						bucket_size_1[t-1][y],
						max_bucket_size_2);
			}
			{
				dim3 block(256, 1);
				dim3 grid(1, 1);
				calc_offset_sum<<<grid, block, 0, stream>>>(
						bucket_offset_2[k].data(),
						bucket_size_2[k].data(),
						num_buckets_2, false);
			}
			{
				dim3 block(1 << HYBRID_SORT_LOG_THREADS, 1);
				dim3 grid(1, num_buckets_2 / 256, 256);
				hybrid_sort_y<<<grid, block, 0, stream>>>(
						PY_tmp[k].data(),
						bucket_size_2[k].data(),
						max_bucket_size_2);
			}

			if(t > final_table)
			{
				dim3 block(128, 1);
				dim3 grid(cdiv(max_bucket_size_2, block.x), num_buckets_2 / 256, 256);

				cuda_memset<uint64_cu>(PD1_out[k], 0, stream);	// TODO: optimize count

				if(t == 3) {
					write_x2<<<grid, block, 0, stream>>>(
							PD1_out[k].data(),
							PD_in[k].data(),
							PY_tmp[k].data(),
							bucket_size_2[k].data(),
							bucket_offset_2[k].data(),
							max_bucket_size_2,
							X2SIZE, XBITS);
				} else {
					write_pd<<<grid, block, 0, stream>>>(
							PD1_out[k].data(),
							PD_in[k].data(),
							PY_tmp[k].data(),
							bucket_size_2[k].data(),
							bucket_offset_2[k].data(),
							max_bucket_size_2);

					if(t > N_TABLE) {
						write_y<<<grid, block, 0, stream>>>(
							Y_out[k].data(),
							PY_tmp[k].data(),
							bucket_size_2[k].data(),
							bucket_offset_2[k].data(),
							max_bucket_size_2);

						if(is_hdd_plot) {
							write_meta<<<grid, block, 0, stream>>>(
								C_out[k].data(),
								C_in[k].data(),
								PY_tmp[k].data(),
								bucket_size_2[k].data(),
								bucket_offset_2[k].data(),
								max_bucket_size_2);
						}
					}
				}
			}

			if(t <= N_TABLE)
			{
				cuda_memset<uint32_t>(num_matches[k], 0, stream);
				{
					dim3 block(128, 1);
					dim3 grid(cdiv(max_bucket_size_2, block.x), num_buckets_2 / 256, 256);
					match_p1<<<grid, block, 0, stream>>>(
							LR_tmp[k].data(),
							t >= 3 ? PD_tmp[k].data() : nullptr,
							num_matches[k].data(),
							PY_tmp[k].data(),
							bucket_size_2[k].data(),
							bucket_offset_2[k].data(),
							num_buckets_2,
							max_bucket_size_2,
							max_bucket_size_1);
				}
				cuda_memset<uint64_cu>(PD_out[k], 0, stream);	// TODO: optimize count
				cuda_memset<uint32_t>(bucket_size_out[k], 0, stream);
				{
					dim3 block(256, 1);
					dim3 grid(max_bucket_size_1 / block.x, 1);
					eval_p1_tx<<<grid, block, 0, stream>>>(
							t == N_TABLE ? Y_out[k].data() : nullptr,
							t < N_TABLE || is_hdd_plot ? C_out[k].data() : nullptr,
							PD_out[k].data(),
							bucket_size_out[k].data(),
							C_in[k].data(),
							t >= 3 ? PD_tmp[k].data() : nullptr,
							t == 2 ? (const uint32_t*)PD_in[k].data() : nullptr,
							LR_tmp[k].data(),
							num_matches[k].data(),
							(uint64_t(y) * max_bucket_size_1) << DSIZE_,
							max_bucket_size_tmp,
							X2SIZE, XBITS, t);
				}
			}
			if(y >= NSTREAMS) {
				wait_barrier(download_barrier[y - NSTREAMS]);
			}

			if(t < N_TABLE) {
				download_buffer(C_buf[k], C_out[k], max_entries_tmp * META_BYTES, stream);
			} else if(is_hdd_plot) {
				download_buffer(C_buf[k], C_out[k], (t <= N_TABLE ? max_entries_tmp : max_bucket_size_1) * META_BYTES_OUT, stream);
			}
			if(t >= N_TABLE) {
				download_buffer(Y_buf[k], Y_out[k], (t <= N_TABLE ? max_entries_tmp : max_bucket_size_1) * YBYTES, stream);
			}
			if(t <= N_TABLE) {
				if(t == 2) {
					download_buffer(PD_buf[k], PD_out[k], max_entries_tmp * X2BYTES, stream);
				} else {
					download_buffer(PD_buf[k], PD_out[k], max_entries_tmp * PDBYTES, stream);
				}
				download_buffer(bucket_size_buf[k], bucket_size_out[k], stream);
			}
			if(t == 3) {
				download_buffer(PD1_buf[k], PD1_out[k], max_entries_tmp * X2BYTES, stream);
			} else if(t > 3) {
				download_buffer(PD1_buf[k], PD1_out[k], max_entries_tmp * PDBYTES, stream);
			}
			cuda_check(cudaEventRecord(download_event[y], stream));

			signal_barrier(stream_barrier[y]);
		}

		for(int k = 0; k < NSTREAMS; ++k) {
			upload_thread[k].join();
			download_thread[k].join();
		}

		flush_buckets(Y_buckets[dst]);
		flush_buckets(C_buckets[dst]);
		flush_buckets(PD_buckets[dst]);
		flush_buckets(PD1_buckets[t-1]);

		delete_buckets(Y_buckets[src]);
		delete_buckets(C_buckets[src]);
		delete_buckets(PD_buckets[src]);

		if(file_store) {
			file_store->flush();
		}
		if(file_store2) {
			file_store2->flush();
		}
		const double elapsed = (get_time_millis() - tx_begin) / 1e3;

		for(int i = 0; i < num_buckets_1; ++i) {
			if(bucket_size_recv[i] > max_bucket_size_1) {
				std::cerr << "WARN: max_bucket_size_1 overflow" << std::endl;
			}
			bucket_size_1[t][i] = std::min<uint32_t>(bucket_size_recv[i], max_bucket_size_1);
		}
		for(int i = 0; i < num_buckets_1; ++i) {
			const auto size = bucket_size_1[std::min(t, N_TABLE)][i];
			result.num_entries += size;
			result.max_bucket_size = std::max(result.max_bucket_size, size);
		}
		result.num_bytes_upload = g_upload_bytes;
		result.num_bytes_download = g_download_bytes;

		if(t <= N_TABLE) {
			std::cout << "[P1] Table " << t;
		} else {
			std::cout << "[P1] Y" << N_TABLE << " sort";
		}
		std::cout << " took " << elapsed << " sec, "
				<< result.num_entries << " entries, " << result.max_bucket_size << " max, " << result.max_bucket_size_tmp << " tmp, "
				<< g_upload_bytes / elapsed / pow(1024, 3) << " GB/s up, "
				<< g_download_bytes / elapsed / pow(1024, 3) << " GB/s down" << std::endl;
		g_upload_bytes = 0;
		g_download_bytes = 0;
	}

	delete [] Y_buf;
	delete [] C_buf;
	delete [] Y_out;
	delete [] Y_in;
	delete [] C_in;
//	delete [] Y_tmp;
	delete [] C_out;
	delete [] PD_in;
	delete [] PD_out;
	delete [] PD_buf;
	delete [] PD1_out;
	delete [] PD1_buf;

	delete [] num_matches;
	delete [] bucket_size_2;
	delete [] bucket_offset_2;
	delete [] PY_tmp;
	delete [] LR_tmp;
	delete [] PD_tmp;

	std::cout << "Phase 1 took " << (get_time_millis() - p1_begin) / 1e3 << " sec" << std::endl;
}

