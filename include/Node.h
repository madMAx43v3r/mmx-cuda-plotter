/*
 * Node.h
 *
 *  Created on: Oct 7, 2021
 *      Author: mad
 */

#ifndef INCLUDE_MMX_NODE_H_
#define INCLUDE_MMX_NODE_H_

#include <util.h>
#include <config.h>
#include <Buffer.h>
#include <Bucket.h>
#include <FileBucket.h>

#include <mmx/PlotHeader.hxx>

#ifndef _WIN32
#include <mad/DirectFile.h>
#endif

#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <stdexcept>
#include <iostream>
#include <cmath>
#include <array>
#include <vector>
#include <queue>
#include <algorithm>


struct input_t {
	int xbits = 0;
	bool ssd_mode = false;
	bool have_contract = false;
	std::array<uint8_t, 32> id = {};
	std::array<uint8_t, 32> seed = {};
	std::array<uint8_t, 33> farmer_key = {};
	std::array<uint8_t, 32> contract = {};
	std::string plot_name;
	std::string tmp_dir;
	std::string tmp_dir2 = "@RAM";
	std::string tmp_dir3 = "@RAM";
};

struct output_t {
	input_t params;
	uint64_t plot_size = 0;
	std::string plot_file_name;
};

struct table_result_t {
	uint32_t max_bucket_size = 0;
	uint32_t max_bucket_size_tmp = 0;
	uint64_t num_entries = 0;
	uint64_t num_bytes_upload = 0;
	uint64_t num_bytes_download = 0;
};


class Node {
private:
	struct ans_tables_t {
		Buffer<uint2> sym_table;
		Buffer<uint16_t> state_table;
	};

public:
	const int device = 0;
	const int num_devices = 0;
	const int final_table = 2;
	const int NSTREAMS = 4;

	Node(int device, int num_devices, int final_table, int NSTREAMS = 4);

	~Node();

	output_t make_plot(const input_t& params);

	void flush();

private:
	struct barrier_t {
		bool flag = false;
		std::condition_variable signal;
	};

	void phase1();
	void phase2();
	void phase3();
	void phase4();

	void await_flush();

	void write_phase4(const std::string& file_name, const uint64_t num_c2_entries, const uint64_t num_c3_parks);

	BucketBase** alloc_buckets(FileStore* file_store = nullptr, uint32_t ram_buckets = 0, bool use_pinned = true);

	void flush_buckets(BucketBase** buckets);

	void delete_buckets(BucketBase**& buckets);

	void sync_chunk(	BucketBase** X_out, const void* X_in,
						const uint32_t* bucket_offset, const uint32_t* bucket_size_in,
						const uint32_t max_bucket_size, const uint32_t max_bucket_size_in, const uint32_t entry_bytes, const uint32_t i);

	void sync_chunk_single(	BucketBase** X_out, const void* X_in,
							const uint32_t* bucket_offset, const uint32_t* bucket_size_in,
							const uint32_t max_bucket_size, const uint32_t max_bucket_size_in, const uint32_t entry_bytes);

	std::vector<uint32_t> update_chunk(std::atomic<uint32_t>* bucket_size, const uint32_t* bucket_size_in, const uint32_t max_bucket_size_in);

	template<typename T, typename S>
	void upload_buffer(Buffer<T>& dst, const Buffer<S>& src, cudaStream_t stream, const size_t count, const size_t offset = 0);

	template<typename T>
	void download_buffer(Buffer<T>& dst, const Buffer<T>& src, const size_t num_bytes, cudaStream_t stream);

	template<typename T>
	void download_buffer(Buffer<T>& dst, const Buffer<T>& src, cudaStream_t stream);

	void reset_barriers();

	inline bool wait_barrier(barrier_t* barrier);
	inline void signal_barrier(barrier_t* barrier);

private:
	input_t params;
	output_t output;

	int hybrid_mode = 0;	// 1 = 256G, 2 = disk

	bool is_hdd_plot = true;

	int XBITS = 0;
	int X2BYTES = 0;
	int X2SIZE = 0;
	int LPX2SIZE = 0;

	int park_size_x = 2048;
	int park_size_y = 8192;
	int park_size_pd = 2048;
	int park_size_meta = 256;

	uint64_t num_buckets_1 = 0;
	uint64_t num_buckets_2 = 0;
	uint64_t max_bucket_size_1 = 0;
	uint64_t max_bucket_size_2 = 0;
	uint64_t max_bucket_size_tmp = 0;
	uint64_t max_entries_tmp = 0;

	Buffer<uint32_t> plot_enc_key;
	Buffer<uint32_t> plot_enc_key_in;
	Buffer<uint32_t> bucket_size_1[N_TABLE + 1];
	Buffer<std::atomic<uint32_t>> bucket_size_recv;

	BucketBase** Y_buckets[2] = {};
	BucketBase** C_buckets[2] = {};
	BucketBase** PD_buckets[2] = {};
	BucketBase** PD1_buckets[N_TABLE + 1] = {};

	std::shared_ptr<FileStore> file_store;
	std::shared_ptr<FileStore> file_store2;

	Buffer<uint32_t>* bucket_size_buf = nullptr;
	Buffer<uint32_t>* bucket_size_out = nullptr;

	Bucket* park_buffer_p7 = nullptr;
	Buffer<uint8_t> park_buffer_c1;
	Buffer<uint8_t> park_buffer_c2;
	Bucket* park_buffer_c3 = nullptr;

	std::vector<cudaStream_t> streams;
	std::vector<cudaStream_t> copy_stream;

	std::vector<cudaEvent_t> sync_event;
	std::vector<cudaEvent_t> sync_event2;
	std::vector<cudaEvent_t> upload_event;
	std::vector<cudaEvent_t> download_event;

	std::vector<barrier_t*> stream_barrier;
	std::vector<barrier_t*> upload_barrier;
	std::vector<barrier_t*> download_barrier;

	std::vector<std::thread> upload_thread;
	std::vector<std::thread> download_thread;

	FILE* plot_file = nullptr;
	std::shared_ptr<mmx::PlotHeader> header;

	std::mutex cout_mutex;
	std::mutex sync_mutex;
	std::mutex flush_mutex;
	std::condition_variable flush_signal;

	struct write_data_t {
		uint64_t offset = 0;
		Bucket* data = nullptr;
	};

	std::mutex write_mutex;
	std::condition_variable write_signal;
	std::queue<write_data_t> write_queue;

	std::vector<std::thread> write_threads;

#ifndef _WIN32
	std::shared_ptr<mad::DirectFile> direct;
#endif

	bool write_done = false;
	bool is_flushed = true;

	std::atomic<uint64_t> g_upload_bytes {0};
	std::atomic<uint64_t> g_download_bytes {0};

};


inline
BucketBase** Node::alloc_buckets(FileStore* file_store, uint32_t ram_buckets, bool use_pinned)
{
	auto out = new BucketBase*[num_buckets_1];
	for(uint32_t i = 0; i < num_buckets_1; ++i) {
		if(!file_store || i < ram_buckets) {
			out[i] = new Bucket(use_pinned);
		} else {
			out[i] = new FileBucket(file_store);
		}
	}
	return out;
}

inline
void Node::flush_buckets(BucketBase** buckets)
{
	if(buckets) {
		for(uint32_t i = 0; i < num_buckets_1; ++i) {
			buckets[i]->flush();
		}
	}
}

inline
void Node::delete_buckets(BucketBase**& buckets)
{
	if(buckets) {
		for(uint32_t i = 0; i < num_buckets_1; ++i) {
			delete buckets[i];
		}
		delete [] buckets;
		buckets = nullptr;
	}
}

template<typename T, typename S>
void copy_single(T* dst, const S* src, const size_t count)
{
	for(size_t i = 0; i < count; ++i) {
		dst[i] = src[i];
	}
}

template<typename T, typename S>
void copy_buffer_single(Buffer<T>& dst, const Buffer<S>& src, const size_t count, const size_t offset = 0)
{
	if(dst.size() < count) {
		throw std::logic_error("dst.size() < count");
	}
	if(src.size() < offset + count) {
		throw std::logic_error("src.size() < offset + count");
	}
	copy_single(dst.data(), src.data(offset), count);
}

template<typename T, typename S>
void copy_buffer_parfor(Buffer<T>& dst, const Buffer<S>& src, const size_t count, const size_t offset, const int nthreads)
{
	if(dst.size() < count) {
		throw std::logic_error("dst.size() < count");
	}
	if(src.size() < offset + count) {
		throw std::logic_error("src.size() < offset + count");
	}
	copy_parfor(dst.data(), src.data(offset), count, nthreads);
}

inline
void Node::sync_chunk(	BucketBase** X_out, const void* X_in,
						const uint32_t* bucket_offset, const uint32_t* bucket_size_in,
						const uint32_t max_bucket_size, const uint32_t max_bucket_size_in, const uint32_t entry_bytes, const uint32_t i)
{
	const auto dst_offset = bucket_offset[i];
	if(dst_offset < max_bucket_size) {
		const auto num_entries = std::min<uint32_t>(
				std::min<uint32_t>(bucket_size_in[i], max_bucket_size_in), max_bucket_size - dst_offset);
		X_out[i]->copy(
				((uint8_t*)X_in) + (uint64_t(i) * max_bucket_size_in) * entry_bytes,
				uint64_t(dst_offset) * entry_bytes, uint64_t(num_entries) * entry_bytes);
	}
}

inline
void Node::sync_chunk_single(	BucketBase** X_out, const void* X_in,
								const uint32_t* bucket_offset, const uint32_t* bucket_size_in,
								const uint32_t max_bucket_size, const uint32_t max_bucket_size_in, const uint32_t entry_bytes)
{
	for(uint32_t i = 0; i < num_buckets_1; ++i) {
		sync_chunk(X_out, X_in, bucket_offset, bucket_size_in, max_bucket_size, max_bucket_size_in, entry_bytes, i);
	}
}

template<typename T, typename S>
void Node::upload_buffer(Buffer<T>& dst, const Buffer<S>& src, cudaStream_t stream, const size_t count, const size_t offset)
{
	if(src.size() < offset + count) {
		throw std::logic_error("src.size() < offset + count");
	}
	if(!src.is_pinned()) {
		throw std::logic_error("src buffer not pinned");
	}
	cuda_check(cudaMemcpyAsync(dst.data(), src.data(offset), count * sizeof(T), cudaMemcpyHostToDevice, stream));
	g_upload_bytes += count * sizeof(T);
}

template<typename T>
void Node::download_buffer(Buffer<T>& dst, const Buffer<T>& src, const size_t num_bytes, cudaStream_t stream)
{
	if(dst.num_bytes() < num_bytes) {
		throw std::logic_error("dst.num_bytes() < num_bytes");
	}
	cuda_check(cudaMemcpyAsync(dst.data(), src.data(), num_bytes, cudaMemcpyDeviceToHost, stream));
	g_download_bytes += num_bytes;
}

template<typename T>
void Node::download_buffer(Buffer<T>& dst, const Buffer<T>& src, cudaStream_t stream)
{
	download_buffer(dst, src, src.num_bytes(), stream);
}

bool Node::wait_barrier(barrier_t* barrier)
{
	bool did_wait = false;
	std::unique_lock<std::mutex> lock(sync_mutex);
	while(!barrier->flag) {
		did_wait = true;
		barrier->signal.wait(lock);
	}
	return did_wait;
}

void Node::signal_barrier(barrier_t* barrier) {
	{
		std::lock_guard<std::mutex> lock(sync_mutex);
		barrier->flag = true;
	}
	barrier->signal.notify_all();
}


#endif /* INCLUDE_MMX_NODE_H_ */
