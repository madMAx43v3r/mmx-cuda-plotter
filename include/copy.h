/*
 * copy.h
 *
 *  Created on: Jun 8, 2021
 *      Author: mad
 */

#ifndef INCLUDE_MMX_COPY_H_
#define INCLUDE_MMX_COPY_H_

#include <stdiox.hpp>

#include <string>
#include <vector>
#include <mutex>
#include <stdexcept>

#include <cstdio>
#include <cstdint>
#include <cstring>
#include <errno.h>


inline
uint64_t copy_file(const std::string& src_path, const std::string& dst_path, const size_t chunk_size)
{
	FILE* src = fopen(src_path.c_str(), "rb");
	if(!src) {
		throw std::runtime_error("fopen() failed for " + src_path + " (" + std::string(std::strerror(errno)) + ")");
	}
	FILE* dst = fopen(dst_path.c_str(), "wb");
	if(!dst) {
		const auto err = errno;
		fclose(src);
		throw std::runtime_error("fopen() failed for " + dst_path + " (" + std::string(std::strerror(err)) + ")");
	}
	uint64_t total_bytes = 0;
	std::vector<uint8_t> buffer(chunk_size);
	while(true) {
		const auto num_bytes = fread(buffer.data(), 1, buffer.size(), src);

		if(fwrite(buffer.data(), 1, num_bytes, dst) != num_bytes) {
			const auto err = errno;
			fclose(src);
			fclose(dst);
			throw std::runtime_error("fwrite() failed on " + dst_path + " (" + std::string(std::strerror(err)) + ")");
		}
		total_bytes += num_bytes;
		if(num_bytes < buffer.size()) {
			break;
		}
	}
	fclose(src);

	if(fclose(dst)) {
		throw std::runtime_error("fclose() failed on " + dst_path + " (" + std::string(std::strerror(errno)) + ")");
	}
	return total_bytes;
}

inline
uint64_t copy_file_direct(const std::string& src_path, const std::string& dst_path, const size_t chunk_size)
{
#ifdef _WIN32
	return copy_file(src_path, dst_path, chunk_size);
#else
	int src = -1;
	for(int i = 0; i < 2 && src < 0; ++i) {
		src = OPEN(src_path.c_str(), O_RDONLY | (i == 0 ? O_DIRECT : 0));
	}
	if(src < 0) {
		throw std::runtime_error("open('" + src_path + "') failed with: " + std::string(std::strerror(errno)));
	}
	int dst = -1;
	for(int i = 0; i < 2 && dst < 0; ++i) {
		dst = OPEN(dst_path.c_str(), O_CREAT | O_WRONLY | (i == 0 ? O_DIRECT : 0), 00644);
	}
	if(dst < 0) {
		const auto err = errno;
		CLOSE(src);
		throw std::runtime_error("open('" + dst_path + "') failed with: " + std::string(std::strerror(err)));
	}
	uint64_t total_bytes = 0;

	std::vector<uint8_t> buffer(chunk_size + 4096);
	uint8_t* p_buf = buffer.data();
	p_buf += (4096 - (size_t(p_buf) % 4096)) % 4096;
	while(true) {
		const auto num_bytes = READ(src, p_buf, chunk_size);
		if(num_bytes < 0) {
			throw std::runtime_error("read() failed on " + src_path + " (" + std::string(std::strerror(errno)) + ")");
		}
		const auto extra_bytes = (4096 - (size_t(num_bytes) % 4096)) % 4096;
		if(extra_bytes) {
			::memset(p_buf + num_bytes, 0, extra_bytes);
		}
		const auto write_bytes = num_bytes + extra_bytes;

		if(WRITE(dst, p_buf, write_bytes) != ssize_t(write_bytes)) {
			const auto err = errno;
			CLOSE(src);
			CLOSE(dst);
			throw std::runtime_error("write() failed on " + dst_path + " (" + std::string(std::strerror(err)) + ")");
		}
		total_bytes += num_bytes;
		if(size_t(num_bytes) < chunk_size) {
			break;
		}
	}
	CLOSE(src);

	if(CLOSE(dst)) {
		throw std::runtime_error("close() failed on " + dst_path + " (" + std::string(std::strerror(errno)) + ")");
	}
	return total_bytes;
#endif
}

inline
uint64_t final_copy(const std::string& src_path, const std::string& dst_path,
					const bool direct_io = true, const size_t chunk_size = 128 * 1024 * 1024)
{
	if(src_path == dst_path) {
		return 0;
	}
	const std::string tmp_dst_path = dst_path + ".tmp";

	uint64_t total_bytes = 0;
	if(rename(src_path.c_str(), tmp_dst_path.c_str())) {
		// try manual copy
		if(direct_io) {
			total_bytes = copy_file_direct(src_path, tmp_dst_path, chunk_size);
		} else {
			total_bytes = copy_file(src_path, tmp_dst_path, chunk_size);
		}
	}
	remove(src_path.c_str());
	rename(tmp_dst_path.c_str(), dst_path.c_str());
	return total_bytes;
}

#ifdef _WIN32
inline
std::string get_socket_error_text() {
	return std::to_string(WSAGetLastError());
}
#else
std::string get_socket_error_text() {
	return std::string(std::strerror(errno)) + " (" + std::to_string(errno) + ")";
}
#endif

inline
::sockaddr_in get_sockaddr_byname(const std::string& endpoint, int port)
{
	::sockaddr_in addr;
	::memset(&addr, 0, sizeof(addr));
	addr.sin_family = AF_INET;
	addr.sin_port = htons(port);
	{
		static std::mutex mutex;
		std::lock_guard<std::mutex> lock(mutex);
		::hostent* host = ::gethostbyname(endpoint.c_str());
		if(!host) {
			throw std::runtime_error("could not resolve: '" + endpoint + "'");
		}
		::memcpy(&addr.sin_addr.s_addr, host->h_addr_list[0], host->h_length);
	}
	return addr;
}

inline
void recv_bytes(void* dst, const int fd, const size_t num_bytes)
{
	auto num_left = num_bytes;
	auto* dst_= (char*)dst;
	while(num_left > 0) {
		const auto num_read = ::recv(fd, dst_, num_left, 0);
		if(num_read < 0) {
			throw std::runtime_error("recv() failed with: " + get_socket_error_text());
		} else if(num_read == 0) {
			throw std::runtime_error("recv() failed with: EOF");
		}
		num_left -= num_read;
		dst_ += num_read;
	}
}

inline
void send_bytes(const int fd, const void* src, const size_t num_bytes)
{
	const auto ret = ::send(fd, (const char*)src, num_bytes, 0);
	if(ret < 0) {
		throw std::runtime_error("send() failed with: " + get_socket_error_text());
	}
	if(size_t(ret) != num_bytes) {
		throw std::runtime_error("send() failed to send all data");
	}
}

inline
uint64_t send_file(	const std::string& src_path, const std::string& dst_host, const int dst_port,
					const bool direct_io = true, const size_t chunk_size = 128 * 1024 * 1024)
{
	int src = -1;
	for(int i = 0; i < 2 && src < 0; ++i) {
		src = OPEN(src_path.c_str(), O_RDONLY | O_BINARY | (i == 0 && direct_io ? O_DIRECT : 0));
	}
	if(src < 0) {
		throw std::runtime_error("open('" + src_path + "') failed with: " + std::string(std::strerror(errno)));
	}
	const uint64_t file_size = LSEEK(src, 0, SEEK_END);
	LSEEK(src, 0, SEEK_SET);

	int fd = -1;
	uint64_t total_bytes = 0;
	try {
		fd = ::socket(AF_INET, SOCK_STREAM, 0);
		if(fd < 0) {
			throw std::runtime_error("socket() failed with: " + get_socket_error_text());
		}
		::sockaddr_in addr = get_sockaddr_byname(dst_host, dst_port);
		if(::connect(fd, (::sockaddr*)&addr, sizeof(addr)) < 0) {
			throw std::runtime_error("connect() failed with: " + get_socket_error_text());
		}
		send_bytes(fd, &file_size, 8);
		{
			char ret = -1;
			recv_bytes(&ret, fd, 1);
			if(ret != 1) {
				if(ret == 0) {
					throw std::runtime_error("no space left on destination");
				} else {
					throw std::runtime_error("unknown error on destination");
				}
			}
		}
		std::string file_name;
		{
			const auto pos = src_path.find_last_of("/\\");
			if(pos != std::string::npos) {
				file_name = src_path.substr(pos + 1);
			} else {
				file_name = src_path;
			}
		}
		{
			const uint16_t name_len = file_name.size();
			send_bytes(fd, &name_len, 2);
			send_bytes(fd, file_name.data(), name_len);
		}

		std::vector<uint8_t> buffer(chunk_size + 4096);
		uint8_t* p_buf = buffer.data();
		p_buf += (4096 - (size_t(p_buf) % 4096)) % 4096;
		while(true) {
			const auto num_bytes = READ(src, p_buf, chunk_size);
			if(num_bytes < 0) {
				throw std::runtime_error("read() failed on " + src_path + " (" + std::string(std::strerror(errno)) + ")");
			}
			send_bytes(fd, p_buf, num_bytes);

			total_bytes += num_bytes;
			if(size_t(num_bytes) < chunk_size) {
				break;
			}
		}
	} catch(...) {
		CLOSESOCKET(fd);
		CLOSE(src);
		throw;
	}
	CLOSESOCKET(fd);
	CLOSE(src);

	remove(src_path.c_str());
	return total_bytes;
}







#endif /* INCLUDE_MMX_COPY_H_ */
