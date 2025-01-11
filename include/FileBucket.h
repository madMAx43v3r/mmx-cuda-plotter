/*
 * FileBucket.h
 *
 *  Created on: Aug 25, 2022
 *      Author: mad
 */

#ifndef INCLUDE_MMX_FILEBUCKET_H_
#define INCLUDE_MMX_FILEBUCKET_H_

#include <Bucket.h>
#include <stdiox.hpp>

#include <list>
#include <queue>
#include <mutex>
#include <thread>
#include <atomic>
#include <condition_variable>
#include <cstdio>


class FileStore {
public:
	std::atomic<uint64_t> total_bytes_read {0};
	std::atomic<uint64_t> total_bytes_written {0};

	FileStore(const std::string& file_path, const size_t num_threads, const size_t max_pending, const bool use_direct_io)
		:	use_direct_io(use_direct_io), file_path(file_path), num_threads(num_threads), max_pending(max_pending)
	{
		auto* file = fopen(file_path.c_str(), "wb");
		if(!file) {
			throw std::runtime_error("fopen() failed for " + file_path);
		}
		fclose(file);

		for(size_t i = 0; i < num_threads; ++i) {
			write_threads.emplace_back(&FileStore::write_loop, this);
		}
	}

	FileStore(const FileStore&) = delete;
	FileStore& operator=(const FileStore&) = delete;

	~FileStore() {
		close();
		free_pool();
	}

	void reset_counters()
	{
		total_bytes_read = 0;
		total_bytes_written = 0;
	}

	uint64_t alloc_chunk()
	{
		std::lock_guard<std::mutex> lock(mutex);

		uint64_t chunk = 0;
		if(pool.empty()) {
			chunk = next_offset;
			next_offset += BUCKET_CHUNK_SIZE;
		} else {
			chunk = pool.back();
			pool.pop_back();
		}
		return chunk;
	}

	uint64_t write_chunk(Buffer<uint8_t>* chunk)
	{
		if(use_direct_io) {
#ifndef _WIN32
			if(chunk->size() % FILE_ALIGNMENT) {
				throw std::logic_error("FileStore: chunk size not aligned to 4K");
			}
			if(size_t(chunk->data()) % FILE_ALIGNMENT) {
				throw std::logic_error("FileStore: chunk address not aligned to 4K");
			}
#endif
		}
		const uint64_t offset = alloc_chunk();
		{
			std::unique_lock<std::mutex> lock(mutex);
			while(num_pending >= max_pending) {
				done_signal.wait(lock);
			}
			write_job_t job;
			job.offset = offset;
			job.chunk = chunk;
			write_queue.push(job);
			num_pending++;
			total_bytes_written += chunk->num_bytes();
		}
		write_signal.notify_one();
		return offset;
	}

	void free_chunk(const uint64_t chunk)
	{
		std::lock_guard<std::mutex> lock(mutex);
		pool.push_back(chunk);
	}

	void flush()
	{
		std::unique_lock<std::mutex> lock(mutex);
		while(num_pending > 0) {
			done_signal.wait(lock);
		}
	}

	void free_pool()
	{
		std::lock_guard<std::mutex> lock(mutex);
		next_offset = 0;
		pool.clear();
		std::remove(file_path.c_str());
	}

	void close()
	{
		{
			std::unique_lock<std::mutex> lock(mutex);
			do_run = false;
		}
		write_signal.notify_all();

		for(auto& thread : write_threads) {
			thread.join();
		}
		write_threads.clear();
	}

	::FILE* open_file(const uint64_t offset = 0) const
	{
		auto* file = fopen(file_path.c_str(), "rb+");
		if(!file) {
			throw std::runtime_error("open('" + file_path + "') failed with: " + std::string(std::strerror(errno)));
		}
		if(offset) {
			if(FSEEK(file, offset, SEEK_SET)) {
				throw std::runtime_error("fseek() failed");
			}
		}
		return file;
	}

	int open_file_direct(const uint64_t offset, const int mode) const
	{
		int file = -1;
		for(int i = 0; i < 2 && file < 0; ++i) {
			file = OPEN(file_path.c_str(), mode | (use_direct_io && i == 0 ? O_DIRECT : 0));
		}
		if(file < 0) {
			throw std::runtime_error("open('" + file_path + "') failed with: " + std::string(std::strerror(errno)));
		}
		LSEEK(file, offset, SEEK_SET);
		return file;
	}

	std::string get_file_path() const {
		return file_path;
	}

private:
	struct write_job_t {
		uint64_t offset = 0;
		Buffer<uint8_t>* chunk = nullptr;
	};

	void write_loop()
	{
#ifdef _WIN32
		const auto file = open_file();
#else
		const int file = open_file_direct(0, O_WRONLY);
#endif
		while(do_run) try
		{
			write_job_t job;
			{
				std::unique_lock<std::mutex> lock(mutex);
				while(write_queue.empty() && do_run) {
					write_signal.wait(lock);
				}
				if(!do_run) {
					break;
				}
				job = write_queue.front();
				write_queue.pop();
			}
#ifdef _WIN32
			FSEEK(file, job.offset, SEEK_SET);

			if(fwrite(job.chunk->data(), 1, job.chunk->size(), file) != job.chunk->size()) {
				throw std::runtime_error("fwrite('" + file_path + "') failed with: " + std::string(std::strerror(errno)));
			}
#else
			LSEEK(file, job.offset, SEEK_SET);

			if(WRITE(file, job.chunk->data(), job.chunk->size()) != ssize_t(job.chunk->size())) {
				throw std::runtime_error("write('" + file_path + "') failed with: " + std::string(std::strerror(errno)));
			}
#endif
			Bucket::free_chunk(job.chunk);
			{
				std::unique_lock<std::mutex> lock(mutex);
				num_pending--;
			}
			done_signal.notify_all();
		}
		catch(const std::exception& ex) {
			std::cerr << "FileStore: " << ex.what() << std::endl;
			break;
		}

#ifdef _WIN32
		if(fclose(file)) {
			throw std::runtime_error("fclose('" + file_path + "') failed with: " + std::string(std::strerror(errno)));
		}
#else
		if(CLOSE(file)) {
			throw std::runtime_error("close('" + file_path + "') failed with: " + std::string(std::strerror(errno)));
		}
#endif
	}

private:
	std::mutex mutex;
	std::condition_variable write_signal;
	std::condition_variable done_signal;

	bool do_run = true;
	bool use_direct_io = false;
	uint64_t next_offset = 0;
	size_t num_pending = 0;
	std::vector<uint64_t> pool;
	std::queue<write_job_t> write_queue;
	std::list<std::thread> write_threads;

	const std::string file_path;
	const size_t num_threads;
	const size_t max_pending;

};


class FileBucket : public BucketBase {
public:
	FileBucket(FileStore* store_) : store(store_) {}

	~FileBucket() {
		free();
	}

	// NEEDS TO BE THREAD SAFE
	size_t upload(void* dst, cudaStream_t stream, const size_t count, const size_t src_offset = 0) override
	{
		std::lock_guard<std::mutex> lock(mutex);
		if(buffer) {
			delete buffer;
		}
		buffer = new Bucket(true, true);

		read_direct(buffer);

		return buffer->upload(dst, stream, count, src_offset);
	}

	// NEEDS TO BE THREAD SAFE
	void copy(const void* src, const size_t dst_offset, const size_t count) override
	{
		if(count == 0) {
			return;
		}
		const auto* p_src = (const uint8_t*)src;
		const auto chunk_offset = dst_offset / BUCKET_CHUNK_SIZE;
		{
			std::unique_lock<std::mutex> lock(mutex);
			// order writes so we only need one cache chunk
			while(dst_offset > num_bytes) {
				signal.wait(lock);
			}
			if(dst_offset < num_bytes) {
				throw std::logic_error("FileBucket: dst_offset < num_bytes");
			}
			size_t left = count;
			for(size_t i = 0; left > 0; ++i)
			{
				const auto off = (i ? 0 : dst_offset % BUCKET_CHUNK_SIZE);
				const auto len = std::min(left, BUCKET_CHUNK_SIZE - off);
				::memcpy(get_cache(chunk_offset + i)->data(off), p_src, len);
				left -= len;
				p_src += len;
			}
			num_bytes += count;
		}
		signal.notify_all();
	}

	void flush() override
	{
		for(size_t i = 0; i < NCACHE; ++i) {
			if(auto& chunk = write_cache[i]) {
				chunks.push_back(store->write_chunk(chunk));
				chunk = nullptr;
			}
		}
	}

	size_t read(void* dst) const
	{
		auto* file = store->open_file();

		auto* dst_ = (uint8_t*)dst;
		auto left = num_bytes;
		for(auto chunk : chunks) {
			FSEEK(file, chunk, SEEK_SET);
			const auto len = std::min(left, BUCKET_CHUNK_SIZE);
			if(fread(dst_, 1, len, file) != len) {
				throw std::runtime_error("fread('" + store->get_file_path() + "') failed with: " + std::string(std::strerror(errno)));
			}
			dst_ += len;
			left -= len;
		}
		if(fclose(file)) {
			throw std::runtime_error("fclose('" + store->get_file_path() + "') failed with: " + std::string(std::strerror(errno)));
		}
		store->total_bytes_read += num_bytes;
		return num_bytes;
	}

	size_t read(Bucket* buffer) const
	{
		auto* file = store->open_file();

		auto left = num_bytes;
		for(auto chunk : chunks) {
			FSEEK(file, chunk, SEEK_SET);
			const auto len = std::min(left, BUCKET_CHUNK_SIZE);
			if(fread(buffer->alloc_next(len), 1, len, file) != len) {
				throw std::runtime_error("fread('" + store->get_file_path() + "') failed with: " + std::string(std::strerror(errno)));
			}
			left -= len;
		}
		if(fclose(file)) {
			throw std::runtime_error("fclose('" + store->get_file_path() + "') failed with: " + std::string(std::strerror(errno)));
		}
		store->total_bytes_read += num_bytes;
		return num_bytes;
	}

	size_t read_direct(void* dst) const
	{
#ifdef _WIN32
		return read(dst);
#else
		const auto file = store->open_file_direct(0, O_RDONLY);

		auto* dst_ = (uint8_t*)dst;
		auto left = num_bytes;
		for(auto chunk : chunks) {
			LSEEK(file, chunk, SEEK_SET);
			auto len = std::min(align_to(left, FILE_ALIGNMENT), BUCKET_CHUNK_SIZE);
			while(len) {
				const auto ret = READ(file, dst_, len);
				if(ret <= 0) {
					throw std::runtime_error("read('" + store->get_file_path() + "') failed with: " + std::string(std::strerror(errno)));
				}
				dst_ += ret;
				len -= ret;
			}
			left -= len;
		}
		if(CLOSE(file)) {
			throw std::runtime_error("close('" + store->get_file_path() + "') failed with: " + std::string(std::strerror(errno)));
		}
		store->total_bytes_read += num_bytes;
		return num_bytes;
#endif
	}

	size_t read_direct(Bucket* buffer) const
	{
#ifdef _WIN32
		return read(buffer);
#else
		const auto file = store->open_file_direct(0, O_RDONLY);

		auto left = num_bytes;
		for(auto chunk : chunks) {
			LSEEK(file, chunk, SEEK_SET);
			auto len = std::min(align_to(left, FILE_ALIGNMENT), BUCKET_CHUNK_SIZE);
			while(len) {
				const auto ret = READ(file, buffer->alloc_next(len), len);
				if(ret <= 0) {
					throw std::runtime_error("read('" + store->get_file_path() + "') failed with: " + std::string(std::strerror(errno)));
				}
				len -= ret;
			}
			left -= len;
		}
		if(CLOSE(file)) {
			throw std::runtime_error("close('" + store->get_file_path() + "') failed with: " + std::string(std::strerror(errno)));
		}
		store->total_bytes_read += num_bytes;
		return num_bytes;
#endif
	}

	void free() override
	{
		for(auto chunk : chunks) {
			store->free_chunk(chunk);
		}
		free_buffer();

		chunks.clear();
		num_bytes = 0;
		write_offset = 0;
	}

	void free_buffer() override
	{
		if(buffer) {
			delete buffer;
			buffer = nullptr;
		}
	}

	size_t size() const override {
		return num_bytes;
	}

private:
	static constexpr size_t NCACHE = 1;

	Buffer<uint8_t>* get_cache(const size_t index)
	{
		if(index < write_offset) {
			throw std::logic_error("FileBucket: index < write_offset");
		}
		while(index - write_offset >= NCACHE)
		{
			if(auto chunk = write_cache[0]) {
				chunks.push_back(store->write_chunk(chunk));
			}
			for(size_t i = 1; i < NCACHE; ++i) {
				write_cache[i - 1] = write_cache[i];
			}
			write_cache[NCACHE - 1] = nullptr;
			write_offset++;
		}
		auto& chunk = write_cache[index - write_offset];
		if(!chunk) {
			chunk = Bucket::alloc_chunk(false, false);
		}
		return chunk;
	}

private:
	std::mutex mutex;
	std::condition_variable signal;
	size_t num_bytes = 0;
	std::vector<uint64_t> chunks;

	size_t write_offset = 0;
	Buffer<uint8_t>* write_cache[NCACHE] = {};

	Bucket* buffer = nullptr;
	FileStore* store = nullptr;

};


#endif /* INCLUDE_MMX_FILEBUCKET_H_ */
