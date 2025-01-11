/*
 * Bucket.h
 *
 *  Created on: Aug 18, 2022
 *      Author: mad
 */

#ifndef INCLUDE_MMX_BUCKET_H_
#define INCLUDE_MMX_BUCKET_H_

#include <Buffer.h>
#include <config.h>
#include <stdiox.hpp>

#ifndef _WIN32
#include <mad/DirectFile.h>
#endif

#include <mutex>
#include <condition_variable>


extern size_t BUCKET_CHUNK_SIZE;

class BucketBase {
public:
	virtual ~BucketBase() {}

	virtual size_t size() const = 0;

	virtual size_t upload(void* dst, cudaStream_t stream, const size_t count, const size_t src_offset = 0) = 0;

	virtual void copy(const void* src, const size_t dst_offset, const size_t count) = 0;

	template<typename T>
	size_t upload(Buffer<T>& dst, cudaStream_t stream)
	{
		return upload(dst.data(), stream, size());
	}

	template<typename T>
	size_t upload(Buffer<T>& dst, cudaStream_t stream, const size_t count, const size_t src_offset = 0, const size_t dst_offset = 0)
	{
		return upload(((uint8_t*)dst.data()) + dst_offset, stream, count == size_t(-1) ? size() : count, src_offset);
	}

	virtual void flush() = 0;
	virtual void free() = 0;
	virtual void free_buffer() = 0;

};

class Bucket : public BucketBase {
public:
	static size_t g_max_pinned_memory;		// bytes

	typedef Buffer<uint8_t> chunk_t;

	Bucket(bool use_pinned = true, bool force_pinned = false)
		: use_pinned(use_pinned), force_pinned(force_pinned)
	{
#ifndef _WIN32
		if(BUCKET_CHUNK_SIZE < FILE_ALIGNMENT) {
			throw std::logic_error("BUCKET_CHUNK_SIZE < FILE_ALIGNMENT");
		}
#endif
	}

	~Bucket() {
		free();
	}

	Bucket(const Bucket&) = delete;
	Bucket& operator=(const Bucket&) = delete;

	using BucketBase::upload;

	// NEEDS TO BE THREAD SAFE
	void copy(const void* src, const size_t dst_offset, const size_t count) override
	{
		const auto* p_src = (const uint8_t*)src;
		const auto chunk_offset = dst_offset / BUCKET_CHUNK_SIZE;

		size_t left = count;
		for(size_t i = 0; left > 0; ++i)
		{
			const auto off = (i ? 0 : dst_offset % BUCKET_CHUNK_SIZE);
			const auto len = std::min(left, BUCKET_CHUNK_SIZE - off);
			::memcpy(get_chunk(chunk_offset + i)->data(off), p_src, len);
			left -= len;
			p_src += len;
		}
		{
			std::lock_guard<std::mutex> lock(mutex);
			num_bytes = std::max(num_bytes, dst_offset + count);
		}
	}

	// NEEDS TO BE THREAD SAFE
	size_t upload(void* dst, cudaStream_t stream, const size_t count, const size_t src_offset_ = 0) override
	{
		size_t src_offset = src_offset_;
		if(src_offset + count > num_bytes) {
			throw std::logic_error("offset + count > num_bytes");
		}
		size_t dst_offset = 0;
		auto* const p_dst = (uint8_t*)dst;

		for(size_t i = src_offset / BUCKET_CHUNK_SIZE; dst_offset < count && i < chunks.size(); ++i)
		{
			const auto off = src_offset % BUCKET_CHUNK_SIZE;
			const auto len = std::min(BUCKET_CHUNK_SIZE - off, count - dst_offset);

			if(auto chunk = chunks[i]) {
				if(chunk->type() == MEM_TYPE_HOST) {
					auto buf = alloc_chunk(true);
					::memcpy(buf->data(), chunk->data(), BUCKET_CHUNK_SIZE);
					{
						std::lock_guard<std::mutex> lock(mutex);
						buf_list.push_back(buf);
					}
					chunk = buf;
				}
				cuda_check(cudaMemcpyAsync(p_dst + dst_offset, chunk->data(off), len, cudaMemcpyHostToDevice, stream));
			} else {
				cuda_check(cudaMemsetAsync(p_dst + dst_offset, 0, len, stream));
			}
			dst_offset += len;
			src_offset += len;
		}
		return count;
	}

	void free_buffer()
	{
		for(auto chunk : buf_list) {
			free_chunk(chunk);
		}
		buf_list.clear();
	}

	size_t fwrite(::FILE* file, const size_t count) const
	{
		size_t offset = 0;
		for(auto chunk : chunks) {
			if(offset >= count) {
				break;
			}
			if(chunk) {
				const auto len = std::min(count - offset, BUCKET_CHUNK_SIZE);
				const auto ret = ::fwrite(chunk->data(), 1, len, file);
				if(size_t(ret) != len) {
					throw std::runtime_error("fwrite() failed with: " + std::string(std::strerror(errno)));
				}
			} else {
				FSEEK(file, BUCKET_CHUNK_SIZE, SEEK_CUR);
			}
			offset += BUCKET_CHUNK_SIZE;
		}
		return offset;
	}

	size_t fwrite(::FILE* file) const
	{
		return fwrite(file, num_bytes);
	}

	size_t write(const int fd, const size_t count) const
	{
		size_t offset = 0;
		for(auto chunk : chunks) {
			if(offset >= count) {
				break;
			}
			if(chunk) {
				const auto len = std::min(count - offset, BUCKET_CHUNK_SIZE);
				const auto ret = WRITE(fd, chunk->data(), len);
				if(size_t(ret) != len) {
					throw std::runtime_error("write() failed with: " + std::string(std::strerror(errno)));
				}
			} else {
				LSEEK(fd, BUCKET_CHUNK_SIZE, SEEK_CUR);
			}
			offset += BUCKET_CHUNK_SIZE;
		}
		return offset;
	}

	size_t write(const int fd) const
	{
		return write(fd, num_bytes);
	}

#ifndef _WIN32
	size_t write_direct(const size_t count, const uint64_t offset, mad::DirectFile& file) const
	{
		mad::DirectFile::buffer_t buffer;

		uint64_t pos = 0;
		for(auto chunk : chunks) {
			if(pos >= count) {
				break;
			}
			if(chunk) {
				const auto len = std::min(count - pos, BUCKET_CHUNK_SIZE);
				file.write(chunk->data(), len, offset + pos, buffer);
			}
			pos += BUCKET_CHUNK_SIZE;
		}
		return pos;
	}

	size_t write_direct(const uint64_t offset, mad::DirectFile& file) const
	{
		return write_direct(num_bytes, offset, file);
	}
#endif

	void align_to(const size_t align)
	{
		const auto aligned_size = ::align_to<size_t>(num_bytes, align);
		if(aligned_size > num_bytes) {
			const std::vector<uint8_t> zeros(aligned_size - num_bytes);
			copy(zeros.data(), num_bytes, zeros.size());
		}
	}

	void flush() override {}

	void free() override
	{
		for(auto chunk : chunks) {
			free_chunk(chunk);
		}
		free_buffer();

		chunks.clear();
		num_bytes = 0;
	}

	size_t size() const override {
		return num_bytes;
	}

	// NOT thread safe
	void* alloc_next(const size_t count)
	{
		if(count > BUCKET_CHUNK_SIZE) {
			throw std::logic_error("alloc_next(): count > BUCKET_CHUNK_SIZE");
		}
		num_bytes += count;
		return get_chunk(chunks.size())->data();
	}

	static chunk_t* alloc_chunk(bool force_pinned = false, bool use_pinned = true)
	{
		std::unique_lock<std::mutex> lock(g_mutex);

try_again:
		if(!force_pinned && !use_pinned && !g_pool_host.empty())
		{
			// use non-pinned if available
			auto chunk = g_pool_host.back();
			g_pool_host.pop_back();
			return chunk;
		}
		if(g_pool.empty() && (g_num_pinned_chunks < g_max_pinned_chunks || force_pinned))
		{
			if(g_block_alloc) {
				g_signal.wait(lock);
				goto try_again;
			}
			// allocate new pinned chunk
			g_num_pinned_chunks++;
			lock.unlock();

			return new chunk_t(BUCKET_CHUNK_SIZE, MEM_TYPE_PINNED);
		}

		chunk_t* chunk = nullptr;
		if(!g_pool.empty() && (g_num_pinned_chunks - g_pool.size() < g_max_pinned_chunks || force_pinned))
		{
			chunk = g_pool.back();
			g_pool.pop_back();
		}
		else if(!g_pool_host.empty())
		{
			chunk = g_pool_host.back();
			g_pool_host.pop_back();
		}
		else {
			if(g_block_alloc) {
				g_signal.wait(lock);
				goto try_again;
			}
			lock.unlock();

			chunk = new chunk_t(BUCKET_CHUNK_SIZE, MEM_TYPE_HOST);
		}
		return chunk;
	}

	static void free_chunk(chunk_t* chunk)
	{
		{
			std::lock_guard<std::mutex> lock(g_mutex);

			if(chunk->type() == MEM_TYPE_PINNED) {
				g_pool.push_back(chunk);
			} else {
				g_pool_host.push_back(chunk);
			}
		}
		g_signal.notify_all();
	}

	static void free_pool()
	{
		std::lock_guard<std::mutex> lock(g_mutex);

		for(auto chunk : g_pool) {
			delete chunk;
		}
		for(auto chunk : g_pool_host) {
			delete chunk;
		}
		g_pool.clear();
		g_pool_host.clear();
	}

	static void init()
	{
		g_max_pinned_chunks = std::min<size_t>(g_max_pinned_memory / BUCKET_CHUNK_SIZE, 60 * 1024);
		g_max_pinned_memory = g_max_pinned_chunks * BUCKET_CHUNK_SIZE;
	}

	static void block_alloc()
	{
		std::lock_guard<std::mutex> lock(g_mutex);
		g_block_alloc = true;
	}

	static void resume_alloc()
	{
		{
			std::lock_guard<std::mutex> lock(g_mutex);
			g_block_alloc = false;
		}
		g_signal.notify_all();
	}

private:
	chunk_t* get_chunk(const size_t index)
	{
		std::lock_guard<std::mutex> lock(mutex);

		if(index < chunks.size()) {
			if(auto chunk = chunks[index]) {
				return chunk;
			}
			return chunks[index] = alloc_chunk(force_pinned, use_pinned);
		}
		chunks.resize(index + 1);
		return chunks[index] = alloc_chunk(force_pinned, use_pinned);
	}

private:
	std::mutex mutex;
	size_t num_bytes = 0;
	bool use_pinned = true;
	bool force_pinned = false;
	std::vector<chunk_t*> chunks;
	std::vector<chunk_t*> buf_list;

	static std::mutex g_mutex;
	static std::condition_variable g_signal;
	static bool g_block_alloc;
	static size_t g_max_pinned_chunks;
	static size_t g_num_pinned_chunks;
	static std::vector<chunk_t*> g_pool;
	static std::vector<chunk_t*> g_pool_host;

};


#endif /* INCLUDE_MMX_BUCKET_H_ */
