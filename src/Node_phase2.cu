/*
 * Node_phase2.cu
 *
 *  Created on: Oct 9, 2021
 *      Author: mad
 */

#include <Node.h>

#include <vnx/vnx.h>

#include <cuda_util.h>
#include <cuda_encoding.h>

#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>


__global__
void mark_used(	uint32_t* B_out, const uint32_t* B_in, const uint64_cu* PD_in,
				const uint32_t bucket_size, const uint32_t max_bucket_size,
				const uint32_t y, const uint64_t max_field_size)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

	if(x >= bucket_size) {
		return;
	}
	const uint64_t i = uint64_t(y) * max_bucket_size + x;
	if(B_in && !(B_in[i / 32] & (uint32_t(1) << (i % 32)))) {
		return;
	}
	const uint64_t PD = read_bits(PD_in, uint64_t(x) * PDSIZE, PDSIZE);
	const uint64_t P_1 = PD >> DSIZE_;
	const uint64_t P_2 = P_1 + 1 + (PD & DMASK);

//	printf("P = %llu, D = %u\n", P_1, 1 + uint32_t(PD & DMASK));

	if(P_1 / 32 < max_field_size) {
		atomicOr(B_out + (P_1 / 32), uint32_t(1) << (P_1 % 32));
	} else {
		printf("mark_used(): P_1 out of range: %llu\n", P_1);
	}
	if(P_2 / 32 < max_field_size) {
		atomicOr(B_out + (P_2 / 32), uint32_t(1) << (P_2 % 32));
	} else {
		printf("mark_used(): P_2 out of range: %llu\n", P_2);
	}
}

__global__
void merge_bitfield(uint32_t* B_out, const uint32_t* B_in, const uint64_t max_field_size)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

	if(x < max_field_size) {
		atomicOr(B_out + x, B_in[x]);
	}
}

__global__
void count_used(uint32_t* count_out, const uint32_t* B_in, const uint64_t max_field_size)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

	count_out[x] = (x < max_field_size ? __popc(B_in[x]) : 0);
}

__device__
uint32_t remap_pos(const uint32_t* B_in, const uint32_t* offset_in, const uint64_t pos)
{
	return offset_in[pos / 32]
			+ __popc(__ldg(B_in + pos / 32) & ((uint32_t(1) << (pos % 32)) - 1));
}

__global__
void remap_pd(	uint64_cu* PD_out, const uint64_cu* PD_in,
				const uint32_t* B_in, const uint32_t* B_next_in,
				const uint32_t* offset_in, const uint32_t* offset_next_in,
				const uint32_t bucket_size, const uint32_t max_bucket_size,
				const uint32_t y, const uint64_t bucket_offset,
				const uint32_t park_size, const uint32_t park_offset,
				const uint64_t max_field_size)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

	if(x >= bucket_size) {
		return;
	}
	const uint64_t i = uint64_t(y) * max_bucket_size + x;
	if(B_in && !(B_in[i / 32] & (uint32_t(1) << (i % 32)))) {
		return;
	}
	const uint64_t PD = read_bits(PD_in, uint64_t(x) * PDSIZE, PDSIZE);
	const uint64_t P_1 = PD >> DSIZE_;
	const uint64_t P_2 = P_1 + 1 + (PD & DMASK);

	uint32_t P_1_new = 0;
	uint32_t P_2_new = 0;
	uint32_t D_new = 0;
	if(P_1 / 32 < max_field_size && P_2 / 32 < max_field_size)
	{
		P_1_new = remap_pos(B_next_in, offset_next_in, P_1);
		P_2_new = remap_pos(B_next_in, offset_next_in, P_2);
		D_new = P_2_new - P_1_new;
	}
	if(D_new >> DSIZE_) {
		printf("remap_pd(): D_new overflow: %d\n", D_new);
	}
	uint64_t pos = bucket_offset + x;
	if(B_in) {
		pos = remap_pos(B_in, offset_in, i);
	}
	const uint32_t park_index = pos / park_size;

	if(park_index >= park_offset) {
		const auto j = (park_index - park_offset) * park_size + pos % park_size;
		if(j < max_bucket_size + park_size) {
			PD_out[j] = (uint64_t(P_1_new) << DSIZE_) | D_new;
		} else {
			printf("remap_pd(): output overflow: %u\n", j);
		}
	} else {
		printf("remap_pd(): park_index < park_offset: %u < %u\n", park_index, park_offset);
	}
}

__global__
void encode_y(	uint64_cu* park_out, const uint32_t* Y_in,
				const uint32_t num_entries, const uint32_t num_parks,
				const uint32_t park_size, const uint32_t park_bytes)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

	if(x >= num_parks) {
		return;
	}
	const uint64_t park_begin = uint64_t(x) * park_bytes * 8;

	uint64_t delta_offset = park_begin + 32;

	uint32_t Y_prev = __ldg(Y_in + x * park_size);
	atomic_write_bits(park_out, Y_prev, park_begin, KSIZE);

	for(uint32_t i = 1; i < park_size; ++i)
	{
		const uint32_t index = x * park_size + i;
		if(index >= num_entries) {
			break;
		}
		const uint32_t Y_i = __ldg(Y_in + index);
		const uint32_t D_i = Y_i - Y_prev;

		if(D_i >> 8) {
			printf("encode_y(): Y delta overflow: %d\n", D_i);
		}
		const uint2 sym = encode_symbol(D_i);

		atomic_write_bits(park_out, sym.x, delta_offset, sym.y);
		delta_offset += sym.y;
		Y_prev = Y_i;
	}
	const uint32_t final_park_bytes = (delta_offset - park_begin) / 8;

	if(final_park_bytes > park_bytes) {
		printf("encode_y(): park overflow: %d > %d\n", final_park_bytes, park_bytes);
	}
}

__global__
void encode_pd(	uint64_cu* park_out, const uint64_cu* PD_in,
				const uint32_t num_entries, const uint32_t num_parks,
				const uint32_t park_size, const uint32_t park_bytes)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

	if(x >= num_parks) {
		return;
	}
	const uint64_t park_begin = uint64_t(x) * park_bytes * 8;

	uint64_t delta_offset = park_begin + park_size * KSIZE;

	for(uint32_t i = 0; i < park_size; ++i)
	{
		const uint32_t index = x * park_size + i;
		if(index >= num_entries) {
			break;
		}
		const uint64_t PD = __ldg(PD_in + index);
		const uint64_t P_i = PD >> DSIZE_;
		const uint8_t  D_i = PD & DMASK;

		const uint2 sym = encode_symbol(D_i);

		atomic_write_bits(park_out, P_i, park_begin + i * KSIZE, KSIZE);
		atomic_write_bits(park_out, sym.x, delta_offset, sym.y);
		delta_offset += sym.y;
	}
	const uint32_t final_park_bytes = (delta_offset - park_begin) / 8;

	if(final_park_bytes > park_bytes) {
		printf("encode_pd(): park overflow: %d > %d\n", final_park_bytes, park_bytes);
	}
}

__global__
void encode_meta(	uint64_cu* park_out, const uint32_t* M_in,
					const uint32_t num_entries, const uint32_t park_size, const uint32_t park_bytes)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

	if(x >= num_entries) {
		return;
	}
	const uint32_t park_index = x / park_size;
	const uint32_t park_offset = x % park_size;
	const uint64_t park_begin = uint64_t(park_index) * park_bytes * 8;

	for(int i = 0; i < N_META_OUT; ++i)
	{
		const auto C_i = M_in[x * N_META_OUT + i];
		atomic_write_bits(park_out, C_i, park_begin + (park_offset * N_META_OUT + i) * KSIZE, KSIZE);
	}
}

// Calculates x * (x-1) / 2. Division is done before multiplication.
template<typename T, typename S>
__device__ __host__
S get_x_enc(const T x)
{
	T a = x;
	T b = x - 1;
	if(a % 2 == 0) {
		a /= 2;
	} else {
		b /= 2;
	}
	return S(a) * b;
}

template<typename T, typename S>
__device__ __host__
S calc_line_point(T x, T y)
{
	return get_x_enc<T, S>(max(x, y)) + min(x, y);
}

template<typename T, typename S>
__device__ __host__
S calc_line_point2(T x, T y)
{
	return calc_line_point<T, S>(x + 1, y + 1);
}

__global__
void remap_x2(	uint2* X2_out, const uint64_cu* X2_in,
				const uint32_t* B_in, const uint32_t* offset_in,
				const uint32_t num_entries, const uint32_t park_size,
				const uint32_t max_bucket_size, const uint32_t y, const uint32_t park_offset,
				const int X2SIZE, const int XBITS)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

	if(x >= num_entries) {
		return;
	}
	const uint64_t i = uint64_t(y) * max_bucket_size + x;
	if(!(B_in[i / 32] & (uint32_t(1) << (i % 32)))) {
		return;
	}
	const uint32_t X_1 = read_bits(X2_in, uint64_t(x) * X2SIZE, XBITS);
	const uint32_t X_2 = read_bits(X2_in, uint64_t(x) * X2SIZE + XBITS, XBITS);

	const uint32_t pos = remap_pos(B_in, offset_in, i);
	const uint32_t park_index = pos / park_size;

	if(park_index >= park_offset) {
		const auto j = (park_index - park_offset) * park_size + pos % park_size;
		X2_out[j] = make_uint2(X_1, X_2);
	} else {
		printf("remap_x2(): park_index < park_offset: %u < %u\n", park_index, park_offset);
	}
}

__global__
void encode_x2(	uint64_cu* park_out, const uint2* X2_in,
				const uint32_t num_entries, const uint32_t park_size,
				const uint32_t park_bytes, const int LPX2SIZE, const int XBITS)
{
	const uint32_t x = blockIdx.x * blockDim.x + threadIdx.x;

	if(x >= num_entries) {
		return;
	}
	const uint32_t park_index = x / park_size;
	const uint32_t park_offset = x % park_size;
	const uint64_t park_begin = uint64_t(park_index) * park_bytes * 8;

	const auto X2 = X2_in[x];

	const uint64_t LPX2 = XBITS < KSIZE ?
			calc_line_point2<uint32_t, uint64_t>(X2.x, X2.y) :
			calc_line_point <uint32_t, uint64_t>(X2.x, X2.y);

	atomic_write_bits(park_out, LPX2, park_begin + park_offset * LPX2SIZE, LPX2SIZE);
}


void Node::phase2()
{
	if(max_bucket_size_1 % 32) {
		throw std::logic_error("max_bucket_size_1 % 32 != 0");
	}
	await_flush();
	{
		// wait for enough space on tmp dir
		const std::experimental::filesystem::path check_path(params.tmp_dir);
		const size_t max_plot_size = (is_hdd_plot ? 320 : 128) * 1.05 * pow(1024, 3) * pow(2.063492, KSIZE - 32);

		bool info_shown = false;
		while(std::experimental::filesystem::space(check_path).available < max_plot_size) {
			if(!info_shown) {
				info_shown = true;
				std::cout << "Waiting for " << max_plot_size / pow(1024, 3) << " GiB available space on: " << params.tmp_dir << std::endl;
			}
			std::this_thread::sleep_for(std::chrono::seconds(1));
		}
	}
	const uint64_t max_field_size = (num_buckets_1 * max_bucket_size_1) / 32;

	const uint64_t max_num_parks_x = cdiv(max_bucket_size_1, park_size_x) + 2;
	const uint64_t max_num_parks_y = cdiv(max_bucket_size_1, park_size_y) + 2;
	const uint64_t max_num_parks_meta = cdiv(max_bucket_size_1, park_size_meta) + 2;
	const uint64_t max_num_parks_pd = cdiv(max_bucket_size_1, park_size_pd) + 2;

	const uint32_t max_park_bytes_x = cdiv(park_size_x * LPX2SIZE, 8);
	const uint32_t max_park_bytes_meta = cdiv(park_size_meta * KSIZE * N_META_OUT, 8);
	const uint32_t max_park_bytes_y = 4 + cdiv(uint32_t((park_size_y - 1) * MAX_AVG_YDELTA_BITS), 8);
	const uint32_t max_park_bytes_pd = cdiv(park_size_pd * KSIZE, 8) + cdiv(uint32_t(park_size_pd * MAX_AVG_OFFSET_BITS), 8);

	const auto p2_begin = get_time_millis();

	Buffer<uint32_t>*  Y_in = new Buffer<uint32_t>[NSTREAMS];
	Buffer<uint32_t>*  C_in = new Buffer<uint32_t>[NSTREAMS];
	Buffer<uint64_cu>* PD_in = new Buffer<uint64_cu>[NSTREAMS];
	Buffer<uint64_cu>* PD_new = new Buffer<uint64_cu>[NSTREAMS];
	Buffer<uint64_cu>* Y_parks = new Buffer<uint64_cu>[NSTREAMS];
	Buffer<uint64_cu>* M_parks = new Buffer<uint64_cu>[NSTREAMS];
	Buffer<uint64_cu>* PD_parks = new Buffer<uint64_cu>[NSTREAMS];
	Buffer<uint64_cu>* Y_buf = new Buffer<uint64_cu>[NSTREAMS];
	Buffer<uint64_cu>* M_buf = new Buffer<uint64_cu>[NSTREAMS];
	Buffer<uint64_cu>* PD_buf = new Buffer<uint64_cu>[NSTREAMS];

	Buffer<uint32_t>* bitfield = new Buffer<uint32_t>[num_devices];
	Buffer<uint32_t>* bitfield_out = new Buffer<uint32_t>[num_devices];
	Buffer<uint32_t>* tmp_count = new Buffer<uint32_t>[num_devices];
	Buffer<uint32_t>* tmp_offset = new Buffer<uint32_t>[num_devices];
	Buffer<uint32_t>* offset_index = new Buffer<uint32_t>[num_devices];

	Buffer<uint32_t> pop_count_host(max_field_size, MEM_TYPE_PINNED);

	for(int i = 0; i < NSTREAMS; ++i) {
		cudaSetDevice(device + i % num_devices);

		if(is_hdd_plot) {
			C_in[i].alloc((max_bucket_size_1 + park_size_meta) * N_META_OUT, MEM_TYPE_DEVICE);
		}
		Y_in[i].alloc(max_bucket_size_1 + park_size_y, MEM_TYPE_DEVICE);
		PD_in[i].alloc((max_bucket_size_1 * PDBYTES) / 8, MEM_TYPE_DEVICE);
		PD_new[i].alloc(max_bucket_size_1 + park_size_pd, MEM_TYPE_DEVICE);

		if(is_hdd_plot) {
			M_parks[i].alloc(cdiv(max_park_bytes_meta * max_num_parks_meta, 8), MEM_TYPE_DEVICE);
			M_buf[i].alloc(M_parks[i].size(), MEM_TYPE_PINNED);
		}
		Y_parks[i].alloc(cdiv(max_park_bytes_y * max_num_parks_y, 8), MEM_TYPE_DEVICE);
		PD_parks[i].alloc(cdiv(max_park_bytes_pd * max_num_parks_pd, 8), MEM_TYPE_DEVICE);
		Y_buf[i].alloc(Y_parks[i].size(), MEM_TYPE_PINNED);
		PD_buf[i].alloc(PD_parks[i].size(), MEM_TYPE_PINNED);
	}

	for(int i = 0; i < num_devices; ++i) {
		cudaSetDevice(device + i);

		bitfield[i].alloc(max_field_size, MEM_TYPE_DEVICE);
		bitfield_out[i].alloc(max_field_size, num_devices > 1 ? MEM_TYPE_MANAGED : MEM_TYPE_DEVICE);
		bitfield_out[i].advise_location(device + i);

		tmp_count[i].alloc(max_field_size, MEM_TYPE_DEVICE);
		tmp_offset[i].alloc(max_field_size, MEM_TYPE_DEVICE);
		offset_index[i].alloc(max_field_size, MEM_TYPE_DEVICE);
	}

	std::cout << "[P2] Setup took " << (get_time_millis() - p2_begin) / 1e3 << " sec" << std::endl;

	header = mmx::PlotHeader::create();
	header->ksize = KSIZE;
	header->xbits = XBITS;
	header->has_meta = is_hdd_plot;
	header->farmer_key = params.farmer_key;
	if(params.have_contract) {
		header->contract = params.contract;
	}
	header->seed = params.seed;
	header->plot_id = params.id;
	header->park_size_x = park_size_x;
	header->park_size_y = park_size_y;
	header->park_size_pd = park_size_pd;
	header->park_size_meta = park_size_meta;
	header->park_bytes_x = max_park_bytes_x;
	header->park_bytes_y = max_park_bytes_y;
	header->park_bytes_pd = max_park_bytes_pd;
	header->park_bytes_meta = max_park_bytes_meta;
	header->entry_bits_x = LPX2SIZE;

	const auto file_name = params.tmp_dir + params.plot_name + ".plot.tmp";
	output.plot_file_name = file_name;

	plot_file = fopen(file_name.c_str(), "wb+");
	if(!plot_file) {
		throw std::runtime_error("fopen('" + file_name + "') failed with: " + std::string(std::strerror(errno)));
	}
	write_done = false;
	is_flushed = false;

	const int num_write_threads = std::max(NSTREAMS, 4);
	write_threads.resize(num_write_threads);

#ifndef _WIN32
	direct = std::make_shared<mad::DirectFile>(file_name, false, true, false);
#endif

	for(int k = 0; k < num_write_threads; ++k) {
		write_threads[k] = std::thread([this, file_name]() {
			try {
				::FILE* file = nullptr;
#ifdef _WIN32
				file = fopen(file_name.c_str(), "rb+");
				if(!file) {
					throw std::runtime_error("fopen('" + file_name + "') failed with: " + std::string(std::strerror(errno)));
				}
#endif
				while(true) {
					std::unique_lock<std::mutex> lock(write_mutex);
					if(!write_queue.empty()) {
						const auto entry = write_queue.front();
//						std::cout << "phase2: writing chunk at offset " << entry.offset << " with size " << entry.data->size() << std::endl;
						write_queue.pop();
						lock.unlock();
						write_signal.notify_all();
#ifndef _WIN32
						entry.data->write_direct(entry.offset, *direct);
#else
						FSEEK(file, entry.offset, SEEK_SET);
						entry.data->fwrite(file);
#endif
						delete entry.data;
					}
					else if(!write_done) {
						write_signal.wait(lock);
					}
					else {
						break;
					}
				}
				if(file) {
					::fclose(file);
				}
			}
			catch(const std::exception& ex) {
				std::lock_guard<std::mutex> lock(cout_mutex);
				std::cerr << "P2 write thread failed with: " << ex.what() << std::endl;
				throw;
			}
		});
	}

	for(int t = N_TABLE; t > final_table; --t)
	{
		const auto tx_begin = get_time_millis();

		const auto src = (t + 1) % 2;

		for(int i = 0; i < num_devices; ++i)
		{
			cudaSetDevice(device + i);
			cuda_memset<uint32_t>(bitfield_out[i], 0, streams[i]);
			cuda_check(cudaEventRecord(sync_event[i], streams[i]));
		}

		reset_barriers();

		// start upload threads
		for(int k = 0; k < NSTREAMS; ++k) {
			upload_thread[k] = std::thread([this, t, k, PD_in]() {
				try {
					for(uint32_t y = k; y < num_buckets_1; y += NSTREAMS)
					{
						if(y >= NSTREAMS) {
							wait_barrier(stream_barrier[y - NSTREAMS]);
						}
						const auto stream = streams[k];

						g_upload_bytes += PD1_buckets[t][y]->upload(PD_in[k], stream);

						cuda_check(cudaEventRecord(upload_event[y], stream));

						signal_barrier(upload_barrier[y]);

						cuda_check(cudaEventSynchronize(upload_event[y]));

						PD1_buckets[t][y]->free_buffer();
					}
				} catch(const std::exception& ex) {
					std::lock_guard<std::mutex> lock(cout_mutex);
					std::cerr << "P2-1 upload thread failed with: " << ex.what() << std::endl;
					throw;
				}
			});
		}

		for(uint32_t y = 0; y < num_buckets_1; ++y)
		{
			const auto k = y % NSTREAMS;
			const auto i = k % num_devices;
			const auto stream = streams[k];

			cudaSetDevice(device + i);

			wait_barrier(upload_barrier[y]);

			cuda_check(cudaStreamWaitEvent(stream, sync_event[i], 0));
			{
				dim3 block(256, 1);
				dim3 grid(max_bucket_size_1 / block.x, 1);
				mark_used<<<grid, block, 0, stream>>>(
						bitfield_out[i].data(),
						t < N_TABLE ? bitfield[i].data() : nullptr,
						PD_in[k].data(),
						bucket_size_1[t][y],
						max_bucket_size_1,
						y, max_field_size);
			}
			signal_barrier(stream_barrier[y]);
		}

		for(int k = 0; k < NSTREAMS; ++k) {
			upload_thread[k].join();
			cuda_check(cudaStreamSynchronize(streams[k]));
		}

		for(int i = 0; i < num_devices; ++i) {
			cudaSetDevice(device + i);

			const auto stream = streams[i];

			// merge bitfields cross device
			for(int k = 0; k < num_devices; ++k) {
				if(k != i) {
					dim3 block(256, 1);
					dim3 grid(max_field_size / block.x, 1);
					merge_bitfield<<<grid, block, 0, stream>>>(
							bitfield_out[i].data(),
							bitfield_out[k].data(),
							max_field_size);
				}
			}
			{
				dim3 block(256, 1);
				dim3 grid(max_field_size / block.x, 1);
				count_used<<<grid, block, 0, stream>>>(
						tmp_count[i].data(),
						bitfield_out[i].data(),
						max_field_size);
			}
			{
				dim3 block(1024, 1);
				dim3 grid(1, 1);
				calc_offset_sum<<<grid, block, 0, stream>>>(
						tmp_offset[i].data(),
						tmp_count[i].data(),
						max_field_size, false);
			}
		}

		for(int i = 0; i < num_devices; ++i) {
			cuda_check(cudaStreamSynchronize(streams[i]));
		}

		const auto time_mid = get_time_millis();
		{
			const double elapsed = (time_mid - tx_begin) / 1e3;
			std::cout << "[P2-1] Table " << t << " took " << elapsed << " sec, "
					<< g_upload_bytes / elapsed / pow(1024, 3) << " GB/s up, "
					<< g_download_bytes / elapsed / pow(1024, 3) << " GB/s down" << std::endl;
			g_upload_bytes = 0;
			g_download_bytes = 0;
		}

		std::vector<uint64_t> num_parks_y(num_buckets_1);
		std::vector<uint64_t> num_parks_pd(num_buckets_1);
		std::vector<uint64_t> num_parks_meta(num_buckets_1);

		std::vector<uint32_t> bucket_size(num_buckets_1);
		std::vector<uint64_t> bucket_offset(num_buckets_1);
		std::vector<uint64_t> bucket_offset_y(num_buckets_1 + 1);
		std::vector<uint64_t> bucket_offset_pd(num_buckets_1 + 1);
		std::vector<uint64_t> bucket_offset_meta(num_buckets_1 + 1);

		for(uint32_t y = 0; y < num_buckets_1; ++y) {
			if(t < N_TABLE) {
				for(uint32_t x = 0; x < max_bucket_size_1 / 32; ++x) {
					bucket_size[y] += pop_count_host[y * max_bucket_size_1 / 32 + x];
				}
			} else {
				bucket_size[y] = bucket_size_1[t][y];
			}
		}
		for(auto& size : bucket_size) {
			size = std::min<uint32_t>(size, max_bucket_size_1);
		}

		uint64_t total_entries = 0;
		{
			for(uint32_t y = 0; y < num_buckets_1; ++y)
			{
				num_parks_y[y] =    (bucket_size[y] + total_entries % park_size_y + (y + 1 < num_buckets_1 ? 0 : park_size_y - 1)) / park_size_y;
				num_parks_pd[y] =   (bucket_size[y] + total_entries % park_size_pd + (y + 1 < num_buckets_1 ? 0 : park_size_pd - 1)) / park_size_pd;
				num_parks_meta[y] = (bucket_size[y] + total_entries % park_size_meta + (y + 1 < num_buckets_1 ? 0 : park_size_meta - 1)) / park_size_meta;

				bucket_offset[y] = total_entries;
				bucket_offset_y[y + 1] = bucket_offset_y[y] + num_parks_y[y] * max_park_bytes_y;
				bucket_offset_pd[y + 1] = bucket_offset_pd[y] + num_parks_pd[y] * max_park_bytes_pd;
				bucket_offset_meta[y + 1] = bucket_offset_meta[y] + num_parks_meta[y] * max_park_bytes_meta;

				total_entries += bucket_size[y];
//				std::cout << "bucket_size[" << y << "] = " << bucket_size[y] << std::endl;
			}
		}
		if(t == N_TABLE) {
			header->num_entries_y = total_entries;
		}
		uint64_t total_parks_y = 0;
		uint64_t total_parks_pd = 0;
		uint64_t total_parks_meta = 0;

		for(uint32_t y = 0; y < num_buckets_1; ++y) {
			total_parks_y += num_parks_y[y];
			total_parks_pd += num_parks_pd[y];
			total_parks_meta += num_parks_meta[y];
		}
		const uint64_t table_size_y = total_parks_y * max_park_bytes_y;
		const uint64_t table_size_pd = total_parks_pd * max_park_bytes_pd;
		const uint64_t table_size_meta = total_parks_meta * max_park_bytes_meta;

		if(t == N_TABLE) {
//			std::cout << "table_size_y = " << table_size_y << std::endl;
//			std::cout << "table_size_meta = " << table_size_meta << std::endl;
			header->table_offset_y = align_to<uint64_t>(4096, FILE_ALIGNMENT);
			const auto table_end_y = align_to(header->table_offset_y + table_size_y, FILE_ALIGNMENT);
			if(is_hdd_plot) {
				header->table_offset_meta = table_end_y;
				header->table_offset_pd.push_back(align_to(header->table_offset_meta + table_size_meta, FILE_ALIGNMENT));
			} else {
				header->table_offset_pd.push_back(table_end_y);
			}
		}
		if(t == final_table + 1) {
			header->table_offset_x = align_to(header->table_offset_pd.back() + table_size_pd, FILE_ALIGNMENT);
		} else {
			header->table_offset_pd.push_back(align_to(header->table_offset_pd.back() + table_size_pd, FILE_ALIGNMENT));
		}
//		std::cout << "table_size_pd = " << table_size_pd << std::endl;

		reset_barriers();

		// start upload threads
		for(int k = 0; k < NSTREAMS; ++k) {
			upload_thread[k] = std::thread([this, t, k, src, Y_in, C_in, PD_in, &bucket_offset]() {
				try {
					for(uint32_t y = k; y < num_buckets_1; y += NSTREAMS)
					{
						if(y >= NSTREAMS) {
							wait_barrier(stream_barrier[y - NSTREAMS]);
						}
						const auto stream = streams[k];
						const auto prev_y = y - 1;
						const auto prev_k = prev_y % NSTREAMS;

						if(y > 0) {
							// wait for copy source to be available
							wait_barrier(upload_barrier[prev_y]);
							cuda_check(cudaStreamWaitEvent(stream, upload_event[prev_y], 0));
						}
						if(y >= NSTREAMS) {
							// prevent over-write of copy source during copy
							const auto prev_next_y = y + 1 - NSTREAMS;
							wait_barrier(upload_barrier[prev_next_y]);
							cuda_check(cudaStreamWaitEvent(stream, upload_event[prev_next_y], 0));
						}
						if(t == N_TABLE) {
							if(is_hdd_plot) {
								const auto prev_count = bucket_offset[y] % park_size_meta;
								if(y > 0) {
									const auto src_offset = (bucket_offset[y] / park_size_meta - bucket_offset[prev_y] / park_size_meta) * park_size_meta * N_META_OUT;
									cuda_check(cudaMemcpyAsync(C_in[k].data(), C_in[prev_k].data(src_offset), prev_count * META_BYTES_OUT, cudaMemcpyDeviceToDevice, stream));
								}
								g_upload_bytes += C_buckets[src][y]->upload(C_in[k], stream, -1, 0, prev_count * META_BYTES_OUT);
							}
							{
								const auto prev_count = bucket_offset[y] % park_size_y;
								if(y > 0) {
									const auto src_offset = (bucket_offset[y] / park_size_y - bucket_offset[prev_y] / park_size_y) * park_size_y;
									cuda_check(cudaMemcpyAsync(Y_in[k].data(), Y_in[prev_k].data(src_offset), prev_count * YBYTES, cudaMemcpyDeviceToDevice, stream));
								}
								g_upload_bytes += Y_buckets[src][y]->upload(Y_in[k], stream, -1, 0, prev_count * YBYTES);
							}
						}
						g_upload_bytes += PD1_buckets[t][y]->upload(PD_in[k], stream);

						cuda_check(cudaEventRecord(upload_event[y], stream));

						signal_barrier(upload_barrier[y]);

						cuda_check(cudaEventSynchronize(upload_event[y]));

						if(t == N_TABLE) {
							if(is_hdd_plot) {
								C_buckets[src][y]->free();
							}
							Y_buckets[src][y]->free();
						}
						PD1_buckets[t][y]->free();
					}
				} catch(const std::exception& ex) {
					std::lock_guard<std::mutex> lock(cout_mutex);
					std::cerr << "P2-2 upload thread failed with: " << ex.what() << std::endl;
					throw;
				}
			});
		}

		// start download threads
		for(int k = 0; k < NSTREAMS; ++k) {
			download_thread[k] = std::thread(
				[this, t, k, PD_buf, Y_buf, M_buf, &bucket_offset_y, &bucket_offset_pd, &bucket_offset_meta]()
			{
				try {
					for(uint32_t y = k; y < num_buckets_1; y += NSTREAMS)
					{
						wait_barrier(stream_barrier[y]);
						cuda_check(cudaEventSynchronize(download_event[y]));

						Bucket* chunk_y = nullptr;
						Bucket* chunk_pd = nullptr;
						Bucket* chunk_meta = nullptr;

						if(t == N_TABLE) {
							if(is_hdd_plot) {
								chunk_meta = new Bucket(false);
								chunk_meta->copy(M_buf[k].data(), 0, bucket_offset_meta[y + 1] - bucket_offset_meta[y]);
							}
							chunk_y = new Bucket(false);
							chunk_y->copy(Y_buf[k].data(), 0, bucket_offset_y[y + 1] - bucket_offset_y[y]);
						}
						chunk_pd = new Bucket(false);
						chunk_pd->copy(PD_buf[k].data(), 0, bucket_offset_pd[y + 1] - bucket_offset_pd[y]);

						signal_barrier(download_barrier[y]);
						{
							std::lock_guard<std::mutex> lock(write_mutex);
							if(chunk_y) {
								write_data_t entry;
								entry.data = chunk_y;
								entry.offset = header->table_offset_y + bucket_offset_y[y];
								write_queue.push(entry);
							}
							if(chunk_meta) {
								write_data_t entry;
								entry.data = chunk_meta;
								entry.offset = header->table_offset_meta + bucket_offset_meta[y];
								write_queue.push(entry);
							}
							if(chunk_pd) {
								write_data_t entry;
								entry.data = chunk_pd;
								entry.offset = header->table_offset_pd[N_TABLE - t] + bucket_offset_pd[y];
								write_queue.push(entry);
							}
						}
						write_signal.notify_all();
					}
				} catch(const std::exception& ex) {
					std::lock_guard<std::mutex> lock(cout_mutex);
					std::cerr << "P2-2 download thread failed with: " << ex.what() << std::endl;
					throw;
				}
			});
		}

		for(uint32_t y = 0; y < num_buckets_1; ++y)
		{
			const auto k = y % NSTREAMS;
			const auto i = k % num_devices;
			const auto stream = streams[k];

			const auto num_entries_y = (y + 1 < num_buckets_1 ? num_parks_y[y] * park_size_y : bucket_size[y] + bucket_offset[y] % park_size_y);
			const auto num_entries_pd = (y + 1 < num_buckets_1 ? num_parks_pd[y] * park_size_pd : bucket_size[y] + bucket_offset[y] % park_size_pd);
			const auto num_entries_meta = (y + 1 < num_buckets_1 ? num_parks_meta[y] * park_size_meta : bucket_size[y] + bucket_offset[y] % park_size_meta);

			const auto park_offset_pd = bucket_offset[y] / park_size_pd;
			const auto prev_count_pd = bucket_offset[y] % park_size_pd;

//			std::cout << y << ": num_parks_y = " << num_parks_y[y] << std::endl;
//			std::cout << y << ": num_parks_pd = " << num_parks_pd[y] << std::endl;
//			std::cout << y << ": num_parks_meta = " << num_parks_meta[y] << std::endl;
//			std::cout << y << ": num_entries_y = " << num_entries_y << std::endl;
//			std::cout << y << ": num_entries_pd = " << num_entries_pd << std::endl;
//			std::cout << y << ": num_entries_meta = " << num_entries_meta << std::endl;
//			std::cout << y << ": park_offset_pd = " << park_offset_pd << std::endl;
//			std::cout << y << ": prev_count_pd = " << prev_count_pd << std::endl;

			cudaSetDevice(device + k % num_devices);

			if(y == 2 * NSTREAMS) {
				Bucket::block_alloc();	// prevent further RAM usage
			}
			wait_barrier(upload_barrier[y]);

			if(y >= NSTREAMS) {
				// prevent over-write of copy source during copy
				cuda_check(cudaStreamWaitEvent(stream, sync_event2[y + 1 - NSTREAMS], 0));
			}
			{
				dim3 block(256, 1);
				dim3 grid(max_bucket_size_1 / block.x, 1);
				remap_pd<<<grid, block, 0, stream>>>(
						PD_new[k].data(),
						PD_in[k].data(),
						t < N_TABLE ? bitfield[i].data() : nullptr,
						bitfield_out[i].data(),
						t < N_TABLE ? offset_index[i].data() : nullptr,
						tmp_offset[i].data(),
						bucket_size_1[t][y],
						max_bucket_size_1, y,
						bucket_offset[y],
						park_size_pd, park_offset_pd,
						max_field_size);
			}
			cuda_check(cudaEventRecord(sync_event[y], stream));

			if(y > 0) {
				const auto prev_y = y - 1;
				const auto prev_k = prev_y % NSTREAMS;
				const auto prev_park_offset_pd = bucket_offset[prev_y] / park_size_pd;

				// wait for copy source to be available
				cuda_check(cudaStreamWaitEvent(stream, sync_event[prev_y], 0));
				cuda_check(cudaMemcpyAsync(
						PD_new[k].data(), PD_new[prev_k].data((park_offset_pd - prev_park_offset_pd) * park_size_pd),
						prev_count_pd * sizeof(uint64_cu), cudaMemcpyDeviceToDevice, stream));
			}
			cuda_check(cudaEventRecord(sync_event2[y], stream));

			cuda_memset<uint64_cu>(PD_parks[k], 0, stream);
			{
				dim3 block(256, 1);
				dim3 grid(cdiv(num_parks_pd[y], block.x), 1);
				encode_pd<<<grid, block, 0, stream>>>(
						PD_parks[k].data(),
						PD_new[k].data(),
						num_entries_pd,
						num_parks_pd[y],
						park_size_pd, max_park_bytes_pd);
			}
			if(t == N_TABLE) {
				cuda_memset<uint64_cu>(Y_parks[k], 0, stream);
				{
					dim3 block(256, 1);
					dim3 grid(cdiv(num_parks_y[y], block.x), 1);
					encode_y<<<grid, block, 0, stream>>>(
							Y_parks[k].data(),
							Y_in[k].data(),
							num_entries_y,
							num_parks_y[y],
							park_size_y, max_park_bytes_y);
				}
				if(is_hdd_plot) {
					cuda_memset<uint64_cu>(M_parks[k], 0, stream);

					dim3 block(256, 1);
					dim3 grid(cdiv(num_entries_meta, block.x), 1);
					encode_meta<<<grid, block, 0, stream>>>(
							M_parks[k].data(),
							C_in[k].data(),
							num_entries_meta,
							park_size_meta, max_park_bytes_meta);
				}
			}

			if(y >= NSTREAMS) {
				wait_barrier(download_barrier[y - NSTREAMS]);
			}
			if(t == N_TABLE) {
				if(is_hdd_plot) {
					download_buffer(M_buf[k], M_parks[k], stream);
				}
				download_buffer(Y_buf[k], Y_parks[k], stream);
			}
			download_buffer(PD_buf[k], PD_parks[k], stream);

			cudaEventRecord(download_event[y], stream);

			if(y >= NSTREAMS) {
				// wait for previous copy to finish before uploading new data
				cuda_check(cudaStreamWaitEvent(stream, sync_event2[y + 1 - NSTREAMS], 0));
			}
			signal_barrier(stream_barrier[y]);
		}

		for(int k = 0; k < NSTREAMS; ++k) {
			upload_thread[k].join();
			download_thread[k].join();
		}
		Bucket::block_alloc();	// prevent further RAM usage

		for(int i = 0; i < num_devices; ++i) {
			cuda_check(cudaMemcpyAsync(bitfield[i].data(), bitfield_out[i].data(), bitfield[i].num_bytes(), cudaMemcpyDeviceToDevice, streams[i]));
			cuda_check(cudaMemcpyAsync(offset_index[i].data(), tmp_offset[i].data(), offset_index[i].num_bytes(), cudaMemcpyDeviceToDevice, streams[i]));
		}
		download_buffer(pop_count_host, tmp_count[0], streams[0]);

		if(t == N_TABLE) {
			delete_buckets(Y_buckets[src]);
			delete_buckets(C_buckets[src]);
		}
		delete_buckets(PD1_buckets[t]);

		for(int i = 0; i < num_devices; ++i) {
			cuda_check(cudaStreamSynchronize(streams[i]));
		}

		const double elapsed = (get_time_millis() - time_mid) / 1e3;
		std::cout << "[P2-2] Table " << t << " took " << elapsed << " sec, "
				<< total_entries << " entries, "
				<< g_upload_bytes / elapsed / pow(1024, 3) << " GB/s up, "
				<< g_download_bytes / elapsed / pow(1024, 3) << " GB/s down" << std::endl;
		g_upload_bytes = 0;
		g_download_bytes = 0;
	}

	delete [] Y_in;
	delete [] C_in;
	delete [] PD_in;
	delete [] PD_new;
	delete [] Y_parks;
	delete [] M_parks;
	delete [] PD_parks;
	delete [] Y_buf;
	delete [] M_buf;
	delete [] PD_buf;
	delete [] tmp_count;
	delete [] tmp_offset;
	delete [] bitfield_out;

	const auto final_begin = get_time_millis();
	{
		Buffer<uint64_cu>* X_in = new Buffer<uint64_cu>[NSTREAMS];
		Buffer<uint2>*     X_tmp = new Buffer<uint2>[NSTREAMS];
		Buffer<uint64_cu>* X_parks = new Buffer<uint64_cu>[NSTREAMS];
		Buffer<uint64_cu>* X_buf = new Buffer<uint64_cu>[NSTREAMS];

		for(int i = 0; i < NSTREAMS; ++i) {
			cudaSetDevice(device + i % num_devices);

			X_in[i].alloc(cdiv(max_bucket_size_1 * X2BYTES, 8), MEM_TYPE_DEVICE);
			X_tmp[i].alloc(max_bucket_size_1 + park_size_x, MEM_TYPE_DEVICE);
			X_parks[i].alloc(cdiv(max_park_bytes_x * max_num_parks_x, 8), MEM_TYPE_DEVICE);
			X_buf[i].alloc(X_parks[i].size(), MEM_TYPE_PINNED);
		}

		std::vector<uint32_t> bucket_size(num_buckets_1);
		std::vector<uint64_t> num_parks_x(num_buckets_1);
		std::vector<uint64_t> bucket_offset(num_buckets_1);
		std::vector<uint64_t> bucket_offset_x(num_buckets_1 + 1);

		for(uint32_t y = 0; y < num_buckets_1; ++y) {
			for(uint32_t x = 0; x < max_bucket_size_1 / 32; ++x) {
				bucket_size[y] += pop_count_host[y * max_bucket_size_1 / 32 + x];
			}
		}
		uint64_t total_entries = 0;
		{
			for(uint32_t y = 0; y < num_buckets_1; ++y)
			{
				num_parks_x[y] = (bucket_size[y] + total_entries % park_size_x + (y + 1 < num_buckets_1 ? 0 : park_size_x - 1)) / park_size_x;

				bucket_offset[y] = total_entries;
				bucket_offset_x[y + 1] = bucket_offset_x[y] + num_parks_x[y] * max_park_bytes_x;

				total_entries += bucket_size[y];
//				std::cout << "bucket_size[" << y << "] = " << bucket_size[y] << std::endl;
			}
		}

		uint64_t total_parks_x = 0;
		for(uint32_t y = 0; y < num_buckets_1; ++y) {
			total_parks_x += num_parks_x[y];
		}
		header->plot_size = header->table_offset_x + total_parks_x * max_park_bytes_x;

		reset_barriers();

		// start upload threads
		for(int k = 0; k < NSTREAMS; ++k) {
			upload_thread[k] = std::thread([this, k, X_in]() {
				try {
					for(uint32_t y = k; y < num_buckets_1; y += NSTREAMS)
					{
						if(y >= NSTREAMS) {
							wait_barrier(stream_barrier[y - NSTREAMS]);
						}
						const auto stream = streams[k];

						g_upload_bytes += PD1_buckets[final_table][y]->upload(X_in[k], stream);

						cuda_check(cudaEventRecord(upload_event[y], stream));

						signal_barrier(upload_barrier[y]);

						cuda_check(cudaEventSynchronize(upload_event[y]));

						PD1_buckets[final_table][y]->free();
					}
				} catch(const std::exception& ex) {
					std::lock_guard<std::mutex> lock(cout_mutex);
					std::cerr << "P2-F upload thread failed with: " << ex.what() << std::endl;
					throw;
				}
			});
		}

		// start download threads
		for(int k = 0; k < NSTREAMS; ++k) {
			download_thread[k] = std::thread([this, k, X_buf, &bucket_offset_x]()
			{
				try {
					for(uint32_t y = k; y < num_buckets_1; y += NSTREAMS)
					{
						wait_barrier(stream_barrier[y]);
						cuda_check(cudaEventSynchronize(download_event[y]));

						auto chunk_x = new Bucket(false);
						chunk_x->copy(X_buf[k].data(), 0, bucket_offset_x[y + 1] - bucket_offset_x[y]);

						signal_barrier(download_barrier[y]);
						{
							std::lock_guard<std::mutex> lock(write_mutex);
							if(chunk_x) {
								write_data_t entry;
								entry.data = chunk_x;
								entry.offset = header->table_offset_x + bucket_offset_x[y];
								write_queue.push(entry);
							}
						}
						write_signal.notify_all();
					}
				} catch(const std::exception& ex) {
					std::lock_guard<std::mutex> lock(cout_mutex);
					std::cerr << "P2-F download thread failed with: " << ex.what() << std::endl;
					throw;
				}
			});
		}

		for(uint32_t y = 0; y < num_buckets_1; ++y)
		{
			const auto k = y % NSTREAMS;
			const auto i = k % num_devices;
			const auto stream = streams[k];

			const auto num_entries_x = (y + 1 < num_buckets_1 ? num_parks_x[y] * park_size_x : bucket_size[y] + bucket_offset[y] % park_size_x);

			const auto park_offset_x = bucket_offset[y] / park_size_x;
			const auto prev_count_x = bucket_offset[y] % park_size_x;

			cudaSetDevice(device + k % num_devices);

			wait_barrier(upload_barrier[y]);

			if(y >= NSTREAMS) {
				// prevent over-write of copy source during copy
				cuda_check(cudaStreamWaitEvent(stream, sync_event2[y + 1 - NSTREAMS], 0));
			}
			{
//				void remap_x2(	uint2* X2_out, const uint64_cu* X2_in,
//								const uint32_t* B_in, const uint32_t* offset_in,
//								const uint32_t num_entries, const uint32_t park_size,
//								const uint32_t max_bucket_size, const uint32_t y, const uint32_t park_offset,
//								const int X2SIZE, const int XBITS);
				dim3 block(256, 1);
				dim3 grid(max_bucket_size_1 / block.x, 1);
				remap_x2<<<grid, block, 0, stream>>>(
						X_tmp[k].data(),
						X_in[k].data(),
						bitfield[i].data(),
						offset_index[i].data(),
						bucket_size_1[final_table][y],
						park_size_x,
						max_bucket_size_1, y, park_offset_x,
						X2SIZE, XBITS);
			}
			cuda_check(cudaEventRecord(sync_event[y], stream));

			if(y > 0) {
				const auto prev_y = y - 1;
				const auto prev_k = prev_y % NSTREAMS;
				const auto prev_park_offset_x = bucket_offset[prev_y] / park_size_x;

				// wait for copy source to be available
				cuda_check(cudaStreamWaitEvent(stream, sync_event[prev_y], 0));
				cuda_check(cudaMemcpyAsync(
						X_tmp[k].data(), X_tmp[prev_k].data((park_offset_x - prev_park_offset_x) * park_size_x),
						prev_count_x * sizeof(uint2), cudaMemcpyDeviceToDevice, stream));
			}
			cuda_check(cudaEventRecord(sync_event2[y], stream));

			cuda_memset<uint64_cu>(X_parks[k], 0, stream);
			{
//				void encode_x2(	uint64_cu* park_out, const uint2* X2_in,
//								const uint32_t num_entries, const uint32_t park_size,
//								const uint32_t park_bytes, const int LPX2SIZE, const int XBITS);
				dim3 block(256, 1);
				dim3 grid(cdiv(num_entries_x, block.x), 1);
				encode_x2<<<grid, block, 0, stream>>>(
						X_parks[k].data(),
						X_tmp[k].data(),
						num_entries_x, park_size_x,
						max_park_bytes_x, LPX2SIZE, XBITS);
			}

			if(y >= NSTREAMS) {
				wait_barrier(download_barrier[y - NSTREAMS]);
			}
			download_buffer(X_buf[k], X_parks[k], stream);

			cudaEventRecord(download_event[y], stream);

			signal_barrier(stream_barrier[y]);
		}

		for(int k = 0; k < NSTREAMS; ++k) {
			upload_thread[k].join();
			download_thread[k].join();
		}

		delete_buckets(PD1_buckets[final_table]);

		const double elapsed = (get_time_millis() - final_begin) / 1e3;
		std::cout << "[P2-2] Table " << final_table << " took " << elapsed << " sec, "
				<< total_entries << " entries, "
				<< g_upload_bytes / elapsed / pow(1024, 3) << " GB/s up, "
				<< g_download_bytes / elapsed / pow(1024, 3) << " GB/s down" << std::endl;
		g_upload_bytes = 0;
		g_download_bytes = 0;

		delete [] X_in;
		delete [] X_tmp;
		delete [] X_buf;
		delete [] X_parks;
	}

	delete [] bitfield;
	delete [] offset_index;

//	vnx::pretty_print(std::cout, header);
//	std::cout << std::endl;

	write_done = true;
	write_signal.notify_all();

	output.plot_size = header->plot_size;

	std::cout << "Phase 2 took " << (get_time_millis() - p2_begin) / 1e3 << " sec" << std::endl;
}

