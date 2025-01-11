/*
 * Bucket.cu
 *
 *  Created on: Aug 18, 2022
 *      Author: mad
 */

#include <Bucket.h>


size_t BUCKET_CHUNK_SIZE = 0;

std::mutex Bucket::g_mutex;
std::condition_variable Bucket::g_signal;

bool Bucket::g_block_alloc = false;
std::vector<Bucket::chunk_t*> Bucket::g_pool;
std::vector<Bucket::chunk_t*> Bucket::g_pool_host;

size_t Bucket::g_max_pinned_memory = -1;
size_t Bucket::g_max_pinned_chunks = -1;
size_t Bucket::g_num_pinned_chunks = 0;

