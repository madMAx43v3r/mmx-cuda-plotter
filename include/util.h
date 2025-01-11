/*
 * util.h
 *
 *  Created on: Jun 27, 2021
 *      Author: mad
 */

#ifndef INCLUDE_MMX_UTIL_H_
#define INCLUDE_MMX_UTIL_H_

#include <chrono>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>
#include <atomic>


template<typename Int, typename Int2>
constexpr inline Int cdiv(Int a, Int2 b) { return (a + b - 1) / b; }

template<typename T>
T align_to(T value, const T align) {
	return value + ((align - (value % align)) % align);
}

inline
int64_t get_time_micros() {
	return std::chrono::duration_cast<std::chrono::microseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

inline
int64_t get_time_millis() {
	return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
}

inline
std::vector<uint8_t> hex_to_bytes(const std::string& hex)
{
	std::vector<uint8_t> result;
	for(size_t i = 0; i < hex.length(); i += 2) {
		const std::string byteString = hex.substr(i, 2);
		result.push_back(::strtol(byteString.c_str(), NULL, 16));
	}
	return result;
}

inline
std::string to_hex_string(const void* data, const size_t length, bool big_endian = false, bool lower_case = false)
{
	static const char map_lower[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'a', 'b', 'c', 'd', 'e', 'f'};
	static const char map_upper[] = {'0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F'};
	const char* map = lower_case ? map_lower : map_upper;
	std::string str;
	str.resize(length * 2);
	for(size_t i = 0; i < length; ++i) {
		if(big_endian) {
			str[(length - i - 1) * 2] = map[((const uint8_t*)data)[i] >> 4];
			str[(length - i - 1) * 2 + 1] = map[((const uint8_t*)data)[i] & 0x0F];
		} else {
			str[i * 2] = map[((const uint8_t*)data)[i] >> 4];
			str[i * 2 + 1] = map[((const uint8_t*)data)[i] & 0x0F];
		}
	}
	return str;
}

inline
std::string get_date_string_ex(const char* format, bool UTC = false, int64_t time_secs = -1) {
	::time_t time_;
	if(time_secs < 0) {
		::time(&time_);
	} else {
		time_ = time_secs;
	}
	::tm* tmp;
	if(UTC) {
		tmp = ::gmtime(&time_);
	} else {
		tmp = ::localtime(&time_);
	}
	char buf[256];
	::strftime(buf, sizeof(buf), format, tmp);
	return std::string(buf);
}

inline
void fseek_set(FILE* file, uint64_t offset) {
	if(fseek(file, offset, SEEK_SET)) {
		throw std::runtime_error("fseek() failed");
	}
}

inline
size_t fwrite_ex(FILE* file, const void* buf, size_t length) {
	if(fwrite(buf, 1, length, file) != length) {
		throw std::runtime_error("fwrite() failed");
	}
	return length;
}

inline
size_t fwrite_at(FILE* file, uint64_t offset, const void* buf, size_t length) {
	fseek_set(file, offset);
	fwrite_ex(file, buf, length);
	return length;
}



#endif /* INCLUDE_MMX_UTIL_H_ */
