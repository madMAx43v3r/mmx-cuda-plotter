/*
 * Node.cu
 *
 *  Created on: Oct 7, 2021
 *      Author: mad
 */

#include <Node.h>
#include <Bucket.h>
#include <vnx/vnx.h>

#include <cuda_util.h>


Node::Node(int device, int num_devices, int final_table, int NSTREAMS)
	:	device(device), num_devices(num_devices), final_table(final_table), NSTREAMS(NSTREAMS),
		num_buckets_1(1 << LOGBUCKETS), num_buckets_2(1 << LOGBUCKETS2)
{
	const auto time_begin = get_time_millis();

	if(NSTREAMS < num_devices) {
		throw std::logic_error("NSTREAMS < num_devices");
	}

	for(int i = 0; i < num_devices; ++i) {
		cudaDeviceProp info;
		cudaGetDeviceProperties(&info, device + i);
		if(info.major < 5 || (info.major == 5 && info.minor < 2)) {
			std::cerr << "GPU[" << device + i << "] '" << info.name << "' is not supported! (need to have CC >= 5.2)" << std::endl;
			throw std::runtime_error("unsupported GPU");
		}
		int value = 0;
		cuda_check(cudaDeviceGetAttribute(&value, cudaDevAttrManagedMemory, device + i));
		if(!value && num_devices > 1) {
			std::cerr << "Your system does not support multi-GPU plotting!" << std::endl;
			throw std::runtime_error("no unified memory support");
		}
		if(num_devices > 1) {
			int value = 0;
			cuda_check(cudaDeviceGetAttribute(&value, cudaDevAttrConcurrentManagedAccess, device + i));
			std::cout << "GPU[" << device + i << "] cudaDevAttrConcurrentManagedAccess = " << value << std::endl;
		}
	}

	cuda_check(cudaSetDevice(device));

	sync_event.resize(num_buckets_1);
	sync_event2.resize(num_buckets_1);
	upload_event.resize(num_buckets_1);
	download_event.resize(num_buckets_1);
	stream_barrier.resize(num_buckets_1);
	upload_barrier.resize(num_buckets_1);
	download_barrier.resize(num_buckets_1);

	upload_thread.resize(NSTREAMS);
	download_thread.resize(NSTREAMS);

	streams.resize(NSTREAMS);
	copy_stream.resize(num_devices);

	for(size_t i = 0; i < streams.size(); ++i) {
		cudaSetDevice(device + i % num_devices);
		cuda_check(cudaStreamCreate(&streams[i]));
	}
	for(size_t i = 0; i < copy_stream.size(); ++i) {
		cudaSetDevice(device + i);
		cuda_check(cudaStreamCreate(&copy_stream[i]));
	}
	for(size_t i = 0; i < num_buckets_1; ++i) {
		cudaSetDevice(device + i % num_devices);
		stream_barrier[i] = new barrier_t();
		upload_barrier[i] = new barrier_t();
		download_barrier[i] = new barrier_t();
		cuda_check(cudaEventCreate(&sync_event[i]));
		cuda_check(cudaEventCreate(&sync_event2[i]));
		cuda_check(cudaEventCreateWithFlags(&upload_event[i], cudaEventBlockingSync | cudaEventDisableTiming));
		cuda_check(cudaEventCreateWithFlags(&download_event[i], cudaEventBlockingSync | cudaEventDisableTiming));
	}

	plot_enc_key.alloc(8, MEM_TYPE_PINNED);
	plot_enc_key_in.alloc(8, num_devices > 1 ? MEM_TYPE_MANAGED : MEM_TYPE_DEVICE);
	plot_enc_key_in.advise_read_mostly();

	for(int i = 1; i <= N_TABLE; ++i) {
		bucket_size_1[i].alloc(num_buckets_1, MEM_TYPE_PINNED);
		bucket_size_1[i].memset_cpu(0);
	}
	bucket_size_recv.alloc(num_buckets_1, MEM_TYPE_PINNED);

	bucket_size_buf = new Buffer<uint32_t>[NSTREAMS];
	bucket_size_out = new Buffer<uint32_t>[NSTREAMS];
	for(int i = 0; i < NSTREAMS; ++i) {
		cudaSetDevice(device + i % num_devices);
		bucket_size_buf[i].alloc(num_buckets_1, MEM_TYPE_PINNED);
		bucket_size_out[i].alloc(num_buckets_1, MEM_TYPE_DEVICE);
	}
	std::cout << "Initialization took " << (get_time_millis() - time_begin) / 1e3 << " sec" << std::endl;
}

Node::~Node()
{
	delete [] bucket_size_buf;
	delete [] bucket_size_out;
}

output_t Node::make_plot(const input_t& params_)
{
	params = params_;

	hybrid_mode = 0;
	if(params.tmp_dir2 != "@RAM") {
		if(params.tmp_dir3 != "@RAM") {
			hybrid_mode = 2;
		} else {
			hybrid_mode = 1;
		}
	}
	if(hybrid_mode && !file_store) {
		const auto path = params.tmp_dir2 + "cuda_plot_tmp2_" + std::to_string(get_time_micros()) + ".tmp";
		file_store = std::make_shared<FileStore>(path, NSTREAMS, num_buckets_1, true);
		std::cout << "Created disk buffer " << path << std::endl;
	}
	if(hybrid_mode >= 2 && !file_store2) {
		const auto path = params.tmp_dir3 + "cuda_plot_tmp3_" + std::to_string(get_time_micros()) + ".tmp";
		file_store2 = std::make_shared<FileStore>(path, NSTREAMS, num_buckets_1, true);
		std::cout << "Created disk buffer " << path << std::endl;
	}
	is_hdd_plot = !params.ssd_mode;

	if(params.xbits >= 16) {
		throw std::logic_error("compression level too high");
	}
	XBITS = KSIZE - params.xbits;
	X2BYTES = cdiv(2 * XBITS, 8);
	X2SIZE = X2BYTES * 8;
	LPX2SIZE = (2 * XBITS - 1);

	max_bucket_size_1 = (uint64_t(1) << KSIZE) / num_buckets_1;
	max_bucket_size_1 = (17 * max_bucket_size_1) / 16;
	max_bucket_size_2 = (uint64_t(4) << KSIZE) / num_buckets_1 / num_buckets_2 / 3;

	max_bucket_size_tmp = (size_t(1) << (KSIZE - 2 * LOGBUCKETS));
	max_bucket_size_tmp += std::max(max_bucket_size_tmp / 24, size_t(1536));

	max_entries_tmp = max_bucket_size_tmp * num_buckets_1;

	cuda_check(cudaSetDevice(device));
	{
//		std::cout << "XBITS = " << XBITS << std::endl;
//		std::cout << "KSIZE = " << KSIZE << std::endl;
//		std::cout << "X2SIZE = " << X2SIZE << std::endl;
//		std::cout << "PDSIZE = " << PDSIZE << std::endl;
//		std::cout << "final_table = " << final_table << std::endl;
//		std::cout << "num_buckets_1 = " << num_buckets_1 << std::endl;
//		std::cout << "num_buckets_2 = " << num_buckets_2 << std::endl;
//		std::cout << "max_bucket_size_1 = " << max_bucket_size_1 << std::endl;
//		std::cout << "max_bucket_size_2 = " << max_bucket_size_2 << std::endl;
//		std::cout << "max_bucket_size_tmp = " << max_bucket_size_tmp << std::endl;
	}
	output.params = params;

	if(file_store) {
		file_store->reset_counters();
	}
	if(file_store2) {
		file_store2->reset_counters();
	}

	phase1();
	phase2();

	const auto tbw_1 = output.plot_size * pow(1024, -3);
	const auto tbw_2 = (file_store ? file_store->total_bytes_written.load() : 0) * pow(1024, -3);
	const auto tbw_3 = (file_store2 ? file_store2->total_bytes_written.load() : 0) * pow(1024, -3);

	std::cout << "[TBW] tmp = " << tbw_1 << " GiB, tmp2 = " << tbw_2 << " GiB, tmp3 = " << tbw_3
			<< " GiB, total = " << tbw_1 + tbw_2 + tbw_3 << " GiB" << std::endl;

	return output;
}

void Node::flush()
{
	const auto flush_begin = get_time_millis();

	for(auto& thread : write_threads) {
		thread.join();
	}
#ifndef _WIN32
	direct->close();
	direct = nullptr;
#endif

	FSEEK(plot_file, 0, SEEK_SET);
	{
		std::vector<uint8_t> header_bytes;
		vnx::VectorOutputStream stream(&header_bytes);
		vnx::TypeOutput out(&stream);
		vnx::write(out, header);
		out.flush();
//		std::cout << "header_size = " << header_bytes.size() << std::endl;

		if(fwrite(header_bytes.data(), 1, header_bytes.size(), plot_file) != header_bytes.size()) {
			throw std::runtime_error("failed to write plot header");
		}
	}
	{
		while(fclose(plot_file)) {
			std::cout << "Closing plot file failed with: " << std::strerror(errno) << std::endl;
			std::this_thread::sleep_for(std::chrono::minutes(1));
		}
		plot_file = nullptr;
		std::cout << "Flushing to disk took " << (get_time_millis() - flush_begin) / 1e3 << " sec" << std::endl;
	}
	{
		std::unique_lock<std::mutex> lock(flush_mutex);
		is_flushed = true;
	}
	flush_signal.notify_all();

	Bucket::resume_alloc();
}

void Node::await_flush()
{
	std::unique_lock<std::mutex> lock(flush_mutex);
	while(!is_flushed) {
		std::cout << "Waiting for disk flush to complete ..." << std::endl;
		flush_signal.wait(lock);
	}
}

std::vector<uint32_t> Node::update_chunk(std::atomic<uint32_t>* bucket_size, const uint32_t* bucket_size_in, const uint32_t max_bucket_size_in)
{
	std::vector<uint32_t> offset(num_buckets_1);
	for(uint32_t i = 0; i < num_buckets_1; ++i) {
		if(i < num_buckets_1) {
			offset[i] = bucket_size[i].fetch_add(std::min<uint32_t>(bucket_size_in[i], max_bucket_size_in));
		}
	}
	return offset;
}

void Node::reset_barriers()
{
	for(size_t i = 0; i < num_buckets_1; ++i) {
		stream_barrier[i]->flag = false;
		upload_barrier[i]->flag = false;
		download_barrier[i]->flag = false;
	}
}

