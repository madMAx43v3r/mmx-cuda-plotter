/*
 * sort.h
 *
 *  Created on: Jun 22, 2021
 *      Author: mad
 */

#ifndef INCLUDE_SORT_H_
#define INCLUDE_SORT_H_

#include <cuda_runtime.h>


template<typename T>
__device__
void insertion_sort(const int x, const int N, const int M, T* buffer)
{
	for(int i = 1; i < N; i++)
	{
		const auto key = buffer[i * M + x];

		int j;
		for(j = i - 1; j >= 0 && buffer[j * M + x] > key; --j) {
			buffer[(j + 1) * M + x] = buffer[j * M + x];
		}
		buffer[(j + 1) * M + x] = key;
	}
}



#endif /* INCLUDE_SORT_H_ */
