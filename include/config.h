/*
 * config.h
 *
 *  Created on: Oct 7, 2021
 *      Author: mad
 */

#ifndef INCLUDE_MMX_CONFIG_H_
#define INCLUDE_MMX_CONFIG_H_

#include <util.h>


#ifndef KSIZE
constexpr int KSIZE = 32;
#endif

#ifndef LOGBUCKETS
constexpr int LOGBUCKETS = 8;
#endif

constexpr int MEM_HASH_N = 32;
constexpr int MEM_HASH_ITER = 256;

constexpr int LOGBUCKETS2 = KSIZE - LOGBUCKETS - 9;

constexpr int N_META = 14;
constexpr int N_META_OUT = 12;
constexpr int N_TABLE = 9;

constexpr int META_BYTES = N_META * 4;
constexpr int META_BYTES_OUT = N_META_OUT * 4;

constexpr int MERGE_SORT_LOG_THREADS = 6;
constexpr int HYBRID_SORT_LOG_THREADS = 6;

constexpr int KBYTES = 4;
constexpr int YBYTES = 4;

constexpr int PSIZE_ = KSIZE + 1;				// position
constexpr int DSIZE_ = 5;
constexpr int PDBYTES = cdiv(PSIZE_ + DSIZE_, 8);
constexpr int PDSIZE = PDBYTES * 8;

constexpr uint32_t DMASK = ((uint64_t(1) << DSIZE_) - 1);
constexpr uint32_t KMASK = ((uint64_t(1) << KSIZE) - 1);

constexpr uint64_t FILE_ALIGNMENT = 4096;

constexpr double MAX_AVG_OFFSET_BITS = 2.65;
constexpr double MAX_AVG_YDELTA_BITS = 2.25;


#endif /* INCLUDE_MMX_CONFIG_H_ */
