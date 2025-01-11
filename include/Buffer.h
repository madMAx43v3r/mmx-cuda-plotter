/*
 * Buffer.h
 *
 *  Created on: Jun 23, 2021
 *      Author: mad
 */

#ifndef INCLUDE_MMX_BUFFER_H_
#define INCLUDE_MMX_BUFFER_H_

#include <vector>
#include <cstring>
#include <cstdint>
#include <cstdlib>

#include <cuda_runtime_util.h>


enum mem_type_e {
	MEM_TYPE_NONE,
	MEM_TYPE_HOST,
	MEM_TYPE_PINNED,
	MEM_TYPE_DEVICE,
	MEM_TYPE_MANAGED,
};


template<typename T>
class Buffer {
public:
	static const size_t MEMORY_ALIGN = 4096;

	Buffer() = default;
	
	Buffer(uint64_t size, mem_type_e type = MEM_TYPE_DEVICE) {
		alloc(size, type);
	}

	Buffer(Buffer& buffer, uint64_t offset, uint64_t size)
		:	Buffer(buffer.data(offset), size, buffer.type_)
	{
	}

	explicit Buffer(T* data, uint64_t size, mem_type_e type)
		:	data_(data), size_(size), type_(type), is_owned_(false)
	{
	}
	
	Buffer(const Buffer&) = delete;
	Buffer& operator=(const Buffer&) = delete;
	
	~Buffer() {
		free();
	}
	
	mem_type_e type() const {
		return type_;
	}

	uint64_t size() const {
		return size_;
	}
	
	uint64_t num_bytes() const {
		return size_ * sizeof(T);
	}
	
	T* data() const {
		return data_;
	}
	
	T* data(uint64_t offset) const {
		return data_ + offset;
	}
	
	T& operator[](uint64_t offset) {
		return data_[offset];
	}
	
	const T& operator[](uint64_t offset) const {
		return data_[offset];
	}
	
	uint8_t* bytes() {
		return (uint8_t*)data_;
	}

	const uint8_t* bytes() const {
		return (const uint8_t*)data_;
	}

	uint8_t* bytes(uint64_t offset) {
		return ((uint8_t*)data_) + offset;
	}

	const uint8_t* bytes(uint64_t offset) const {
		return ((const uint8_t*)data_) + offset;
	}

	void free() {
		if(data_) {
			if(is_registered_) {
				cuda_unregister();
			}
			if(is_owned_) {
				switch(type_) {
					case MEM_TYPE_HOST:
						::free(data_);
						break;
					case MEM_TYPE_PINNED:
						cuda_check(cudaFreeHost(ptr_));
						break;
					default:
						cuda_check(cudaFree(data_));
				}
			}
			ptr_ = nullptr;
			data_ = nullptr;
			type_ = MEM_TYPE_NONE;
		}
	}
	
	void alloc(mem_type_e type = MEM_TYPE_DEVICE) {
		alloc(size_, type);
	}
	
	void alloc(uint64_t size, mem_type_e type = MEM_TYPE_DEVICE)
	{
		free();
		size_ = size;
		if(size) {
			try {
				switch(type) {
					case MEM_TYPE_HOST: {
						const auto num_bytes = size_ * sizeof(T);
#ifdef _WIN32
						data_ = (T*)::malloc(num_bytes);
#else
						data_ = (T*)::aligned_alloc(MEMORY_ALIGN, num_bytes + (MEMORY_ALIGN - num_bytes % MEMORY_ALIGN) % MEMORY_ALIGN);
#endif
						if(!data_) {
							throw std::runtime_error("malloc() / aligned_alloc() returned nullptr");
						}
						break;
					}
					case MEM_TYPE_PINNED: {
						cuda_check(cudaMallocHost(&ptr_, num_bytes()));
						uint8_t* ptr = (uint8_t*)ptr_;
#ifndef _WIN32
						// make sure it's aligned
						if(num_bytes() >= MEMORY_ALIGN && size_t(ptr) % MEMORY_ALIGN)
						{
							cuda_check(cudaFreeHost(ptr_));
							cuda_check(cudaMallocHost(&ptr_, num_bytes() + 4096));
							ptr = (uint8_t*)ptr_;
							ptr += (MEMORY_ALIGN - (size_t(ptr) % MEMORY_ALIGN)) % MEMORY_ALIGN;
						}
#endif
						data_ = (T*)ptr;
						break;
					}
					case MEM_TYPE_DEVICE:
						cuda_check(cudaMalloc((void**)&data_, num_bytes()));
						break;
					case MEM_TYPE_MANAGED:
						cuda_check(cudaMallocManaged((void**)&data_, num_bytes()));
						break;
					default:
						throw std::logic_error("invalid memory type");
				}
			} catch(const std::exception& ex) {
				const auto nbytes = num_bytes();
				size_ = 0;
				std::string type_name;
				switch(type) {
					case MEM_TYPE_HOST: type_name = "MEM_TYPE_HOST"; break;
					case MEM_TYPE_DEVICE: type_name = "MEM_TYPE_DEVICE"; break;
					case MEM_TYPE_PINNED: type_name = "MEM_TYPE_PINNED"; break;
					case MEM_TYPE_MANAGED: type_name = "MEM_TYPE_MANAGED"; break;
					default: type_name = std::to_string(int(type));
				}
				throw std::runtime_error("failed to allocate " + std::to_string(nbytes) + " bytes of " + type_name + ": " + ex.what());
			}
		}
		type_ = type;
	}
	
	void alloc_min(uint64_t size, mem_type_e type = MEM_TYPE_DEVICE)
	{
		if(size > size_ || type != type_) {
			alloc(size, type);
		}
	}

	void wrap(Buffer& buffer, uint64_t offset, uint64_t size) {
		wrap(buffer.data(offset), size, buffer.type_);
	}

	void wrap(T* data, uint64_t size, mem_type_e type)
	{
		free();
		data_ = data;
		size_ = size;
		type_ = type;
		is_owned_ = false;
	}

	void cuda_register() {
		if(!is_registered_) {
			cuda_check(cudaHostRegister(data_, num_bytes(), 0));
			is_registered_ = true;
		}
	}

	void cuda_unregister() {
		if(is_registered_) {
			is_registered_ = false;
			cuda_check(cudaHostUnregister(data_));
		}
	}

	bool is_pinned() const {
		return type_ == MEM_TYPE_PINNED || is_registered_;
	}

	T* malloc_host() const {
		return new T[size_];
	}
	
	void upload(const T* host) {
		cuda_check(cudaMemcpy(data_, host, num_bytes(), cudaMemcpyHostToDevice));
	}
	
	void download(T* host) const {
		cuda_check(cudaMemcpy(host, data_, num_bytes(), cudaMemcpyDeviceToHost));
	}
	
	std::vector<T> download() const {
		std::vector<T> out(size_);
		download(out.data());
		return out;
	}
	
	void copy_from(const Buffer<T>& src) const
	{
		enum cudaMemcpyKind kind = cudaMemcpyDefault;
		switch(type_) {
			case MEM_TYPE_HOST:
			case MEM_TYPE_PINNED:
				switch(src.type_) {
					case MEM_TYPE_DEVICE:
					case MEM_TYPE_MANAGED:
						kind = cudaMemcpyDeviceToHost;
						break;
				}
				break;
			case MEM_TYPE_DEVICE:
			case MEM_TYPE_MANAGED:
				switch(src.type_) {
					case MEM_TYPE_HOST:
					case MEM_TYPE_PINNED:
						kind = cudaMemcpyHostToDevice;
						break;
					case MEM_TYPE_DEVICE:
					case MEM_TYPE_MANAGED:
						kind = cudaMemcpyDeviceToDevice;
						break;
				}
				break;
		}
		if(src.num_bytes() != num_bytes()) {
			throw std::runtime_error("size mismatch");
		}
		cuda_check(cudaMemcpy(data_, src.data(), num_bytes(), kind));
	}
	
	void memset(int value) {
		cuda_check(cudaMemset(data_, value, num_bytes()));
	}
	
	void memset_cpu(int value) {
		::memset(data_, value, num_bytes());
	}

	void memset_async(int value, cudaStream_t stream) {
		cuda_check(cudaMemsetAsync(data_, value, num_bytes(), stream));
	}
	
	void memset_async(int value, uint64_t offset, uint64_t length, cudaStream_t stream) {
		cuda_check(cudaMemsetAsync(data_ + offset, value, length * sizeof(T), stream));
	}
	
	void advise_read_mostly()
	{
		if(type_ == MEM_TYPE_MANAGED) {
			cuda_check(cudaMemAdvise(data_, num_bytes(), cudaMemAdviseSetReadMostly, -1));
		}
	}

	void advise_location(int device, uint64_t length = 0, uint64_t offset = 0)
	{
		if(type_ == MEM_TYPE_MANAGED) {
			if(!length) {
				length = size_;
			}
			int value = 0;
			cuda_check(cudaDeviceGetAttribute(&value, cudaDevAttrConcurrentManagedAccess, device));
			if(value) {
				cuda_check(cudaMemAdvise(data_ + offset, length * sizeof(T), cudaMemAdviseSetPreferredLocation, device));
			}
		}
	}

private:
	uint64_t size_ = 0;
	mem_type_e type_ = MEM_TYPE_NONE;
	void* ptr_ = nullptr;
	T* data_ = nullptr;
	bool is_owned_ = true;
	bool is_registered_ = false;
	
};


template<typename T>
void cuda_memset(Buffer<T>& buffer, const T& value, cudaStream_t stream) {
	cuda_memset<T>(buffer.data(), value, buffer.size(), stream);
}

template<typename T>
void cuda_memset(Buffer<T>& buffer, const T& value, cudaStream_t stream, const size_t count) {
	cuda_memset<T>(buffer.data(), value, count, stream);
}

template<typename T>
void cuda_memcpy(T* dst, Buffer<T>& src, cudaStream_t stream) {
	cuda_memcpy<T>(dst, src.data(), src.size(), stream);
}

template<typename T>
void cuda_memcpy(T* dst, Buffer<T>& src, cudaStream_t stream, const size_t count) {
	cuda_memcpy<T>(dst, src.data(), count, stream);
}


#endif /* INCLUDE_MMX_BUFFER_H_ */
