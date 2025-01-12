/*
 * cuda_plot.cu
 *
 *  Created on: Nov 15, 2021
 *      Author: mad
 */

#include <set>
#include <list>
#include <vector>
#include <string>
#include <csignal>
#include <cmath>

#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>

#include <vnx/vnx.h>
#include <vnx/SHA256.h>
#include <uint256_t.h>

#include <Node.h>
#include <util.h>
#include <copy.h>

#include <cxxopts.hpp>
#include <bech32.h>
#include <version.hpp>

#ifdef __linux__ 
	#include <unistd.h>
	#define GETPID getpid
#elif _WIN32
	#include <processthreadsapi.h>
	#define GETPID GetCurrentProcessId
#else
	#define GETPID() int(-1)
#endif

std::atomic_bool do_exit {false};
std::atomic_bool signal_trigger {false};
std::atomic_bool gracefully_exit {false};

static void interrupt_handler(int sig)
{
    signal_trigger = true;
}

static void exit_handler()
{
	int64_t interrupt_timestamp = 0;
	while(!do_exit) {
		if(signal_trigger) {
			signal_trigger = false;
			if(((get_time_micros() - interrupt_timestamp) / 1e6) <= 1 ) {
				std::cout << std::endl << "Double Ctrl-C pressed, exiting now!" << std::endl;
				exit(-4);
			} else {
				interrupt_timestamp = get_time_micros();
			}
			if(!gracefully_exit) {
				std::cout << std::endl;
				std::cout << "****************************************************************************************" << std::endl;
				std::cout << "**  The crafting of plots will stop after the creation and copy of the current plot.  **" << std::endl;
				std::cout << "**         !! If you want to force quit now, press Ctrl-C twice in series !!          **" << std::endl;
				std::cout << "****************************************************************************************" << std::endl;
				gracefully_exit = true;
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(100));
	}
}

static std::array<uint8_t, 32> bech32_address_decode(const std::string& addr)
{
	const auto res = bech32::decode(addr);
	if(res.encoding != bech32::Bech32m) {
		throw std::logic_error("invalid address: " + addr);
	}
	if(res.dp.size() != 52) {
		throw std::logic_error("invalid address (size != 52): " + addr);
	}
	uint256_t bits = 0;
	for(int i = 0; i < 50; ++i) {
		bits |= res.dp[i] & 0x1F;
		bits <<= 5;
	}
	bits |= res.dp[50] & 0x1F;
	bits <<= 1;
	bits |= (res.dp[51] >> 4) & 1;

	std::array<uint8_t, 32> bytes;
	for(size_t i = 0; i < 32; ++i) {
		bytes[i] = bits & 0xFF;
		bits >>= 8;
	}
	return bytes;
}

static std::array<uint8_t, 32> sha256(const void* data, const size_t num_bytes)
{
	vnx::SHA256 ctx;
	ctx.update((const uint8_t*)data, num_bytes);
	std::array<uint8_t, 32> out;
	ctx.finalize(out.data());
	return out;
}

static
output_t create_plot(	const int k,
						const int xbits,
						const int clevel,
						const bool ssd_mode,
						Node* node,
						const std::array<uint8_t, 32>& contract,
						const std::array<uint8_t, 33>& farmer_key,
						const std::string& tmp_dir,
						const std::string& tmp_dir2,
						const std::string& tmp_dir3)
{
	const auto total_begin = get_time_millis();
	const bool have_puzzle = contract != std::array<uint8_t, 32>();
	
	std::cout << "Process ID: " << GETPID() << std::endl;
	std::cout << "Farmer Key: " << to_hex_string(farmer_key.data(), farmer_key.size()) << std::endl;
	
	if(have_puzzle) {
		std::cout << "Contract Hash: " << to_hex_string(contract.data(), contract.size()) << std::endl;
	}
	std::array<uint8_t, 32> seed;
	vnx::secure_random_bytes(seed.data(), seed.size());
	
	input_t params;
	params.xbits = xbits;
	params.ssd_mode = ssd_mode;
	params.seed = seed;
	params.contract = contract;
	params.farmer_key = farmer_key;
	{
		uint32_t offset = 0;
		uint8_t buf[1024] = {};
		if(have_puzzle) {
			const std::string tag("MMX/PLOTID/NFT");
			::memcpy(buf + offset, tag.data(), tag.size()); offset += tag.size();
		} else {
			const std::string tag("MMX/PLOTID/OG");
			::memcpy(buf + offset, tag.data(), tag.size()); offset += tag.size();
		}
		const uint8_t ksize = k;
		::memcpy(buf + offset, &ksize, 1); offset += 1;
		::memcpy(buf + offset, seed.data(), seed.size()); offset += seed.size();
		::memcpy(buf + offset, farmer_key.data(), farmer_key.size()); offset += farmer_key.size();
		if(have_puzzle) {
			params.have_contract = true;
			params.contract = contract;
			::memcpy(buf + offset, contract.data(), contract.size()); offset += contract.size();
		}
		params.id = sha256(buf, offset);
	}
	std::string prefix = "plot-mmx-" + std::string(ssd_mode ? "ssd" : "hdd");
	const std::string plot_name = prefix + "-k" + std::to_string(k) + "-c" + std::to_string(clevel)
			+ "-" + get_date_string_ex("%Y-%m-%d-%H-%M")
			+ "-" + vnx::to_hex_string(params.id.data(), params.id.size());
	
	std::cout << "Working Directory:   " << (tmp_dir.empty() ? "$PWD" : tmp_dir) << std::endl;
	std::cout << "Working Directory 2: " << tmp_dir2 << std::endl;
	if(tmp_dir3 != "@RAM") {
		std::cout << "Working Directory 3: " << tmp_dir3 << std::endl;
	}
	std::cout << "Compression Level: C" << clevel << " (" << (ssd_mode ? "SSD" : "HDD") << ")" << std::endl;
	std::cout << "Plot Name: " << plot_name << std::endl;
	
	params.plot_name = plot_name;
	params.tmp_dir = tmp_dir;
	params.tmp_dir2 = tmp_dir2;
	params.tmp_dir3 = tmp_dir3;
	
	const auto out = node->make_plot(params);
	
	const auto time_secs = (get_time_millis() - total_begin) / 1e3;
	std::cout << "Total plot creation time was "
			<< time_secs << " sec (" << time_secs / 60. << " min)" << std::endl;
	return out;
}

int _main(int argc, char** argv)
{
#ifdef _WIN32
	{
		WSADATA data;
		const int wsaret = WSAStartup(MAKEWORD(1, 1), &data);
		if(wsaret != 0) {
			std::cerr << "WSAStartup() failed with error: " << wsaret << "\n";
			exit(-1);
		}
	}
#endif

	cxxopts::Options options("mmx_cuda_plot",
		"MMX k" + std::to_string(KSIZE) + " CUDA plotter" +
#ifdef GIT_COMMIT_HASH
		" - " GIT_COMMIT_HASH
#endif
		"\n\n"
		"For <farmerkey> see output of `mmx wallet keys`.\n"
		"To plot for pooling, specify -c <contract> address, see `mmx wallet plotnft show`.\n"
		"In case of <count> != 1, you may press Ctrl-C for graceful termination after current plot is finished,\n"
		"or double press Ctrl-C to terminate immediately.\n\n"
	);
	
	std::string farmer_key_str;
	std::string contract_addr_str;
	std::vector<std::string> tmp_dir;
	std::string tmp_dir2 = "@RAM";
	std::string tmp_dir3 = "@RAM";
	std::vector<std::string> final_out;
	const int k = KSIZE;
	int C = 0;
	int device = 0;
	int num_devices = 1;
	int NSTREAMS = 3;
	int bucket_chunk_size = -1;
	int num_plots = 1;
	int copy_port = 1337;
	int max_tmp_plots = -1;
	int max_total_copy = -1;
	int max_parallel_copy = 1;
	bool ssd_mode = false;
	bool waitforcopy = false;
	bool use_direct_io = true;
	double max_pinned_memory = -1;
	
	options.allow_unrecognised_options().add_options()(
		"C, level", "Compression level (0 to 15)", cxxopts::value<int>(C))(
		"ssd", "Make SSD plots", cxxopts::value<bool>(ssd_mode))(
		"n, count", "Number of plots to create (default = 1, unlimited = -1)", cxxopts::value<int>(num_plots))(
		"g, device", "CUDA device (default = 0)", cxxopts::value<int>(device))(
		"r, ndevices", "Number of CUDA devices (default = 1)", cxxopts::value<int>(num_devices))(
		"t, tmpdir", "Temporary directories for plot storage (default = $PWD)", cxxopts::value<std::vector<std::string>>(tmp_dir))(
		"2, tmpdir2", "Temporary directory 2 for partial RAM / disk mode (default = @RAM)", cxxopts::value<std::string>(tmp_dir2))(
		"3, tmpdir3", "Temporary directory 3 for disk mode (default = @RAM)", cxxopts::value<std::string>(tmp_dir3))(
		"d, finaldir", "Final destinations (default = <tmpdir>, remote = @HOST)", cxxopts::value<std::vector<std::string>>(final_out))(
		"z, dstport", "Destination port for remote copy (default = 1337)", cxxopts::value<int>(copy_port))(
		"w, waitforcopy", "Wait for copy to start next plot", cxxopts::value<bool>(waitforcopy))(
		"c, contract", "Pool Contract Address (62 chars)", cxxopts::value<std::string>(contract_addr_str))(
		"f, farmerkey", "Farmer Public Key (33 bytes)", cxxopts::value<std::string>(farmer_key_str))(
//		"D, directio", "Use direct IO for final copy (default = false, Linux only)", cxxopts::value<bool>(use_direct_io))(
		"S, streams", "Number of parallel streams (default = 3, must be >= 2)", cxxopts::value<int>(NSTREAMS))(
		"B, chunksize", "Bucket chunk size in MiB (default = 16, 1 to 256)", cxxopts::value<int>(bucket_chunk_size))(
		"Q, maxtmp", "Max number of plots to cache in tmpdir (default = -1)", cxxopts::value<int>(max_tmp_plots))(
		"A, copylimit", "Max number of parallel copies in total (default = -1)", cxxopts::value<int>(max_total_copy))(
		"W, maxcopy", "Max number of parallel copies to same HDD (default = 1, unlimited = -1)", cxxopts::value<int>(max_parallel_copy))(
		"M, memory", "Max shared / pinned memory in GiB (default = unlimited)", cxxopts::value<double>(max_pinned_memory))(
		"version", "Print version")(
		"h, help", "Print help");
	
	if(argc <= 1) {
		std::cout << options.help({""}) << std::endl;
		return 0;
	}
	const auto args = options.parse(argc, argv);
	
	if(args.count("help")) {
		std::cout << options.help({""}) << std::endl;
		return 0;
	}
	if(args.count("version")) {
		std::cout << kVersion << std::endl;
		return 0;
	}
	if(farmer_key_str.empty()) {
		std::cout << "Farmer Public Key (33 bytes) needs to be specified via -f, see `mmx wallet keys`." << std::endl;
		return -2;
	}
	if(tmp_dir3 != "@RAM") {
		if(tmp_dir2 == "@RAM") {
			tmp_dir2 = tmp_dir3;
		}
	}
	if(tmp_dir2 != "@RAM") {
		if(tmp_dir.empty()) {
			tmp_dir.push_back(tmp_dir2);
		}
	}
	if(tmp_dir.empty()) {
		std::cout << "tmpdir needs to be specified via -t path/" << std::endl;
		return -2;
	}
	int avail_devices = 0;
	cudaGetDeviceCount(&avail_devices);

	if(device + num_devices > avail_devices) {
		std::cout << "Invalid -r | --ndevices, not enough devices: " << avail_devices << std::endl;
		return -2;
	}
	if(num_devices < 1 || (num_devices & (num_devices - 1))) {
		std::cout << "Invalid -r | --ndevices, needs to be a power of two" << std::endl;
		return -2;
	}
	if(NSTREAMS < 2) {
		std::cout << "-S | --streams needs to be >= 2" << std::endl;
		return -2;
	}
	if(max_tmp_plots == 0) {
		std::cout << "Invalid -Q | --maxtmp, needs to be != 0" << std::endl;
		return -2;
	}
	if(max_total_copy == 0) {
		std::cout << "Invalid -A | --copylimit, needs to be != 0" << std::endl;
		return -2;
	}
	if(max_parallel_copy == 0) {
		std::cout << "Invalid -W | --maxcopy, needs to be != 0" << std::endl;
		return -2;
	}
	if(bucket_chunk_size <= 0) {
		if(tmp_dir2 != "@RAM") {
			if(tmp_dir3 != "@RAM") {
				bucket_chunk_size = 2;	// disk mode
			} else {
				bucket_chunk_size = 4;	// partial RAM
			}
		} else {
			bucket_chunk_size = 16;		// full RAM
		}
	}
	if(bucket_chunk_size > 256) {
		std::cout << "Invalid -B | --chunksize, needs to be <= 256" << std::endl;
		return -2;
	}
	BUCKET_CHUNK_SIZE = size_t(bucket_chunk_size) * 1024 * 1024;

	int xbits = 0;
	int final_table = 0;

	if(C >= 0 && C <= 15) {
		xbits = C;
		final_table = 2;
	} else {
		std::cout << "Invalid compression level: " << C << std::endl;
		return -2;
	}

	std::array<uint8_t, 32> contract = {};
	std::array<uint8_t, 33> farmer_key = {};
	{
		const auto tmp = hex_to_bytes(farmer_key_str);
		if(tmp.size() != farmer_key.size()) {
			std::cout << "Invalid farmer key: '" << farmer_key_str << "' (needs to be " << farmer_key.size() << " bytes)" << std::endl;
			return -2;
		}
		::memcpy(farmer_key.data(), tmp.data(), tmp.size());
	}

	if(!contract_addr_str.empty()) {
		try {
			contract = bech32_address_decode(contract_addr_str);
		}
		catch(std::exception& ex) {
			std::cout << "Invalid contract (address): '" << contract_addr_str
					<< "' (" << ex.what() << ", see `mmx wallet plotnft show`)" << std::endl;
			return -2;
		}
	}
	for(const auto& dir : tmp_dir) {
		if(!dir.empty() && dir.find_last_of("/\\") != dir.size() - 1) {
			std::cout << "Invalid tmpdir: " << dir << " (needs trailing '/' or '\\')" << std::endl;
			return -2;
		}
	}
	if(!tmp_dir2.empty() && tmp_dir2 != "@RAM" && tmp_dir2.find_last_of("/\\") != tmp_dir2.size() - 1) {
		std::cout << "Invalid tmpdir2: " << tmp_dir2 << " (needs trailing '/' or '\\')" << std::endl;
		return -2;
	}
	if(!tmp_dir3.empty() && tmp_dir3 != "@RAM" && tmp_dir3.find_last_of("/\\") != tmp_dir3.size() - 1) {
		std::cout << "Invalid tmpdir2: " << tmp_dir3 << " (needs trailing '/' or '\\')" << std::endl;
		return -2;
	}
	for(const auto& final_dir : final_out)
	{
		if(final_dir.empty()) {
			std::cout << "Invalid destination: empty" << std::endl;
			return -2;
		}
		if(final_dir[0] != '@')
		{
			if(final_dir.find_last_of("/\\") != final_dir.size() - 1) {
				std::cout << "Invalid destination: " << final_dir << " (needs trailing '/' or '\\')" << std::endl;
				return -2;
			}
			const auto prefix = final_dir + char(std::experimental::filesystem::path::preferred_separator);
			// check if this folder is disabled
			if(		!std::experimental::filesystem::exists(prefix + "chia_plot_sink_disable")
				&&	!std::experimental::filesystem::exists(prefix + "chia_plot_sink_disable.txt"))
			{
				const std::string path = final_dir + ".cuda_plot_final";
				if(auto file = fopen(path.c_str(), "wb")) {
					fclose(file);
					remove(path.c_str());
				} else {
					std::cout << "Failed to write to directory: '" << final_dir << "'" << std::endl;
					return -2;
				}
			}
		}
	}
	for(const auto& dir : tmp_dir) {
		const std::string path = dir + ".cuda_plot_tmp";
		if(auto file = fopen(path.c_str(), "wb")) {
			fclose(file);
			remove(path.c_str());
		} else {
			std::cout << "Failed to write to tmpdir directory: '" << dir << "'" << std::endl;
			return -2;
		}
	}
	if(tmp_dir2 != "@RAM") {
		const std::string path = tmp_dir2 + ".cuda_plot_tmp";
		if(auto file = fopen(path.c_str(), "wb")) {
			fclose(file);
			remove(path.c_str());
		} else {
			std::cout << "Failed to write to tmpdir2 directory: '" << tmp_dir2 << "'" << std::endl;
			return -2;
		}
	}
	if(tmp_dir3 != "@RAM") {
		const std::string path = tmp_dir3 + ".cuda_plot_tmp";
		if(auto file = fopen(path.c_str(), "wb")) {
			fclose(file);
			remove(path.c_str());
		} else {
			std::cout << "Failed to write to tmpdir3 directory: '" << tmp_dir3 << "'" << std::endl;
			return -2;
		}
	}

	std::thread exit_thread(&exit_handler);

	if(num_plots > 1 || num_plots < 0) {
		if(std::signal(SIGINT, interrupt_handler) == SIG_ERR) {
			std::cerr << "std::signal(SIGINT) failed!" << std::endl;
		}
		if(std::signal(SIGTERM, interrupt_handler) == SIG_ERR) {
			std::cerr << "std::signal(SIGTERM) failed!" << std::endl;
		}
	}
#ifndef _WIN32
	std::signal(SIGPIPE, SIG_IGN);
#endif
	
	std::cout << "MMX k" + std::to_string(KSIZE) + " CUDA plotter";
	#ifdef GIT_COMMIT_HASH
		std::cout << " - " << GIT_COMMIT_HASH;
	#endif	
	std::cout << std::endl;
	std::cout << std::endl;
	std::cout << "No. GPUs: " << num_devices << std::endl;
	std::cout << "No. Streams: " << NSTREAMS << std::endl;
#ifndef _WIN32
	std::cout << "Direct IO: " << (use_direct_io ? "Yes" : "No") << std::endl;
#endif
	for(const auto& final_dir : final_out) {
		std::cout << "Final Destination: " << final_dir << std::endl;
	}
	std::cout << "Bucket Chunk Size: " << BUCKET_CHUNK_SIZE / 1024 / 1024 << " MiB" << std::endl;

	if(max_pinned_memory >= 0) {
		const auto limit = std::max<double>(max_pinned_memory - 2, 0) * 0.95;
		Bucket::g_max_pinned_memory = limit * pow(1024, 3);
	}
	Bucket::init();

	std::cout << "Max Pinned Memory: " << Bucket::g_max_pinned_memory / pow(1024, 3) << " GiB" << std::endl;

	if(num_plots >= 0) {
		std::cout << "Number of Plots: " << num_plots << std::endl;
	} else {
		std::cout << "Number of Plots: infinite" << std::endl;
	}
	
	auto node = new Node(device, num_devices, final_table, NSTREAMS * num_devices);

	std::mutex mutex;
	std::thread flush_thread;
	std::condition_variable signal;

	bool flush_done = true;
	size_t copy_offset = 0;
	std::set<std::string> dst_failed;
	std::map<std::string, size_t> dst_active;
	std::map<std::string, uint64_t> dst_reserved;
	std::map<std::string, std::thread> copy_threads;

	for(int i = 0; i < num_plots || num_plots < 0; ++i)
	{
		if(gracefully_exit) {
			std::cout << std::endl << "Process has been interrupted, waiting for copy operations to finish ..." << std::endl;
			break;
		}
		if(max_tmp_plots > 0) {
			bool did_warn = false;
			std::unique_lock<std::mutex> lock(mutex);
			while(copy_threads.size() + (flush_done ? 0 : 1) + 1 > size_t(max_tmp_plots)) {
				if(!did_warn) {
					did_warn = true;
					std::cout << "Reached maximum number of plots in tmpdir (" << max_tmp_plots << "), waiting ..." << std::endl;
				}
				signal.wait(lock);
			}
		}
		std::cout << "Crafting plot " << i+1 << " out of " << num_plots
				<< " (" << get_date_string_ex("%Y/%m/%d %H:%M:%S") << ")" << std::endl;

		const auto tmp_dir1 = tmp_dir[i % tmp_dir.size()];

		const auto out = create_plot(
				k, xbits, C, ssd_mode, node, contract, farmer_key, tmp_dir1, tmp_dir2, tmp_dir3);

		if(flush_thread.joinable()) {
			flush_thread.join();
		}
		flush_done = false;

		flush_thread = std::thread([node, &mutex, &signal, &flush_done, &copy_offset, &dst_failed, &dst_active, &dst_reserved, &copy_threads, out, tmp_dir1, final_out, copy_port, max_total_copy, max_parallel_copy, use_direct_io]()
		{
			node->flush();

			std::lock_guard<std::mutex> lock(mutex);

			const std::string src_path = tmp_dir1 + out.params.plot_name + ".plot";

			// start copy
			copy_threads[src_path] = std::thread([&mutex, &signal, &copy_offset, &dst_failed, &dst_active, &dst_reserved, &copy_threads, out, tmp_dir1, src_path, final_out, copy_port, max_total_copy, max_parallel_copy, use_direct_io]()
			{
				while(::rename(out.plot_file_name.c_str(), src_path.c_str())) {
					std::cout << "Rename to " << src_path << " failed with: " << std::strerror(errno) << std::endl;
					std::this_thread::sleep_for(std::chrono::minutes(1));
				}
				uint64_t src_size = 0;
				try {
					src_size = std::experimental::filesystem::file_size(src_path);
				} catch(const std::exception& ex) {
					std::cout << "Warning: Failed to get file size for " << src_path << " (" << ex.what() << ")" << std::endl;
				}

				bool success = false;
				bool showed_waiting = false;
				while(!success && !final_out.empty())
				{
					std::string dst_host;
					std::string dst_path;
					std::string final_dir;
					size_t final_offset = 0;
					{
						std::unique_lock<std::mutex> lock(mutex);

						if(max_total_copy > 0) {
							bool did_warn = false;
							while(true) {
								size_t total = 0;
								for(const auto& entry : dst_active) {
									total += entry.second;
								}
								if(total + 1 > size_t(max_total_copy)) {
									if(!did_warn) {
										did_warn = true;
										std::cout << "Reached maximum number of total parallel copies (" << max_total_copy << "), waiting ..." << std::endl;
									}
									signal.wait(lock);
								} else {
									break;
								}
							}
						}

						std::vector<std::tuple<std::string, size_t, uint64_t, size_t>> dirs;
						for(size_t k = 0; k < final_out.size(); ++k)
						{
							const auto index = (copy_offset + k) % final_out.size();
							const auto& dir = final_out[index];
							const auto num_active = dst_active[dir];
							const bool is_remote = (dir[0] == '@');

							if(!dst_failed.count(dir)) {
								if(is_remote) {
									dirs.emplace_back(dir, num_active, -1, index);
								}
								else if(num_active < size_t(max_parallel_copy)) {
									const auto prefix = dir + char(std::experimental::filesystem::path::preferred_separator);
									// check if this folder is disabled
									if(		!std::experimental::filesystem::exists(prefix + "chia_plot_sink_disable")
										&&	!std::experimental::filesystem::exists(prefix + "chia_plot_sink_disable.txt"))
									{
										const auto available = std::experimental::filesystem::space(dir).available;
										dirs.emplace_back(dir, num_active, available, index);
									}
								}
							}
						}
						std::sort(dirs.begin(), dirs.end(),
							[](const std::tuple<std::string, size_t, uint64_t, size_t>& L, const std::tuple<std::string, size_t, uint64_t, size_t>& R) -> bool {
								return std::get<1>(L) == std::get<1>(R) ? std::get<2>(L) > std::get<2>(R) : std::get<1>(L) < std::get<1>(R);
							});

						for(const auto& entry : dirs) {
							const auto& dir = std::get<0>(entry);
							if(dir[0] == '@') {
								final_dir = dir;
								final_offset = std::get<3>(entry);
								break;
							} else {
								try {
									const auto available = std::get<2>(entry);
									const auto required = src_size + dst_reserved[dir];
									if(available > required) {
										final_dir = dir;
										final_offset = std::get<3>(entry);
										break;
									}
								} catch(const std::exception& ex) {
									std::cout << "Warning: Failed to get free space for " << dir << " (" << ex.what() << ")" << std::endl;
								}
							}
						}
						if(final_dir.empty()) {
							if(!showed_waiting) {
								showed_waiting = true;
								std::cout << "All destinations are busy or full, waiting ..." << std::endl;
							}
							signal.wait_for(lock, std::chrono::seconds(10));
							continue;
						}

						if(final_dir == tmp_dir1) {
							break;
						}
						if(final_dir[0] == '@') {
							dst_host = final_dir.substr(1);
							dst_path = final_dir + ":" + out.params.plot_name + ".plot";
						} else {
							dst_path = final_dir + out.params.plot_name + ".plot";
						}
						dst_active[final_dir]++;
						dst_reserved[final_dir] += src_size;
						copy_offset = final_offset + 1;

						std::cout << "Started copy to " << dst_path << std::endl;
					}
					const auto total_begin = get_time_millis();

					try {
						size_t num_bytes = 0;
						if(dst_host.empty()) {
							num_bytes = final_copy(src_path, dst_path, use_direct_io);
						} else {
							num_bytes = send_file(src_path, dst_host, copy_port, use_direct_io);
						}
						std::lock_guard<std::mutex> lock(mutex);

						const auto time = (get_time_millis() - total_begin) / 1e3;
						if(num_bytes) {
							std::cout << "Copy to " << dst_path << " finished, took " << time << " sec, "
								<< ((num_bytes / time) / 1024 / 1024) << " MB/s" << std::endl;
						} else {
							std::cout << "Renamed final plot to " << dst_path << std::endl;
						}
						success = true;
					}
					catch(const std::exception& ex) {
						std::cout << "Copy to " << dst_path << " failed with: " << ex.what() << std::endl;
					}
					{
						std::lock_guard<std::mutex> lock(mutex);
						if(!success && dst_host.empty()) {
							dst_failed.insert(final_dir);
						}
						dst_active[final_dir]--;
						dst_reserved[final_dir] -= src_size;
					}
					signal.notify_all();

					if(!success) {
						std::this_thread::sleep_for(std::chrono::seconds(60));
					}
				}
				{
					std::lock_guard<std::mutex> lock(mutex);
					auto iter = copy_threads.find(src_path);
					if(iter != copy_threads.end()) {
						iter->second.detach();
						copy_threads.erase(iter);
					}
				}
				signal.notify_all();
			});

			flush_done = true;
			signal.notify_all();
		});

		if(waitforcopy)
		{
			flush_thread.join();

			std::unique_lock<std::mutex> lock(mutex);
			while(!copy_threads.empty()) {
				signal.wait(lock);
			}
		}
	}

	if(flush_thread.joinable()) {
		flush_thread.join();
	}
	delete node;
	Bucket::free_pool();

	{
		std::unique_lock<std::mutex> lock(mutex);
		while(!copy_threads.empty()) {
			signal.wait(lock);
		}
	}
	for(const auto& path : dst_failed) {
		std::cout << "Failed drive: " << path << std::endl;
	}

	do_exit = true;
	exit_thread.join();
	
#ifdef _WIN32
	WSACleanup();
#endif
	return 0;
}

#ifdef _WIN32

int main(int argc, char** argv)
{
	try {
		return _main(argc, argv);
	}
	catch(const std::exception& ex) {
		std::cerr << "Fatal error: " << ex.what() << std::endl;
	}
	return -1;
}

#else

int main(int argc, char** argv)
{
	return _main(argc, argv);
}

#endif /* _WIN32 */
