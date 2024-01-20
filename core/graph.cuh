/*
Copyright (c) 2014-2015 Xiaowei Zhu, Tsinghua University
Copyright (c) 2018 Hippolyte Barraud, Tsinghua University

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
*/

#ifndef GRAPH_H
#define GRAPH_H

#define N ((long)1024*1024*512)
#define BS 256
#define GS (N+BS-1)/BS
#define MAX_EDGES IOSIZE/4

#define WORD_OFFSET(i) (i >> 6)
#define BIT_OFFSET(i) (i & 0x3f)

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <unistd.h>
#ifdef USE_OPENMP
#include <omp.h>
#endif
#include <cstring>

#include <thread>
#include <vector>
#include <functional>

#include "constants.hpp"
#include "type.hpp"
#include "bitmap.hpp"
#include "atomic.hpp"
#include "queue.hpp"
#include "partition.hpp"
#include "bigvector.hpp"
#include "time.hpp"
#include "common.h"

bool f_true(VertexId v) {
	return true;
}

void f_none_1(std::pair<VertexId,VertexId> vid_range) {

}

void f_none_2(std::pair<VertexId,VertexId> source_vid_range, std::pair<VertexId,VertexId> target_vid_range) {

}

template <typename T>
void process(VertexId *edge_h, T *parent_data, unsigned long long int * active_in_data, unsigned long long int * active_out_data, T *local_value_h, int edges) {
	for (int i = 0; i < edges; i++) {
		if (edge_h[i*2] != -1 && edge_h[i*2+1] != -1) {
			if (active_in_data==nullptr || active_in_data[WORD_OFFSET(edge_h[i*2])] & (1ul<<BIT_OFFSET(edge_h[i*2]))) {
				if (parent_data[edge_h[i*2+1]]==-1) {
					if (cas(&parent_data[edge_h[i*2+1]], -1, edge_h[i*2])) {
						__sync_fetch_and_or(active_out_data+WORD_OFFSET(edge_h[i*2+1]), 1ul<<BIT_OFFSET(edge_h[i*2+1]));
						*local_value_h += 1;
					}
				}
			}
		}
	}
	return;
}

template <typename T>
__global__ void process_e(int *edge_d, T *parent_data_d, unsigned long long int *active_in_d, unsigned long long int *active_out_d, T *local_value_d, int edges){
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx >= edges) return;

	// if (edge_d[idx*2] == -1 || edge_d[idx*2+1] == -1) return;

	if (active_in_d==nullptr || active_in_d[WORD_OFFSET(edge_d[idx*2])] & (1ul<<BIT_OFFSET(edge_d[idx*2]))) {
		if (parent_data_d[edge_d[idx*2+1]]==-1) {
			if (atomicCAS(parent_data_d+edge_d[idx*2+1], -1, edge_d[idx*2]) == -1) {
				atomicOr(active_out_d+WORD_OFFSET(edge_d[idx*2+1]), 1ul<<BIT_OFFSET(edge_d[idx*2+1]));
				atomicAdd(local_value_d, 1);
			}
		}
	}
	return;
}

class Graph {
	int parallelism;
	int edge_unit;
	bool * should_access_shard;
	long ** fsize;
	char ** buffer_pool;
	long * column_offset;
	long * row_offset;
	long memory_bytes;
	int partition_batch;
	long vertex_data_bytes;
	long PAGESIZE;
public:
	std::string path;

	int edge_type;
	VertexId vertices;
	EdgeId edges;
	int partitions;

	Graph (std::string path) {
		PAGESIZE = 4096;
		parallelism = std::thread::hardware_concurrency();
		// buffer_pool = new char * [parallelism*1];
		// for (int i=0;i<parallelism*1;i++) {
		// 	buffer_pool[i] = (char *)memalign(PAGESIZE, IOSIZE);
		// 	assert(buffer_pool[i]!=NULL);
		// 	memset(buffer_pool[i], 0, IOSIZE);
		// }
		init(path);
	}

	void set_memory_bytes(long memory_bytes) {
		this->memory_bytes = memory_bytes;
	}

	void set_vertex_data_bytes(long vertex_data_bytes) {
		this->vertex_data_bytes = vertex_data_bytes;
	}

	void init(std::string path) {
		int c;
		this->path = path;

		FILE * fin_meta = fopen((path+"/meta").c_str(), "r");
		c = fscanf(fin_meta, "%d %d %ld %d", &edge_type, &vertices, &edges, &partitions);
		fclose(fin_meta);

		if (edge_type==0) {
			PAGESIZE = 4096;
		} else {
			PAGESIZE = 12288;
		}

		should_access_shard = new bool[partitions];

		if (edge_type==0) {
			edge_unit = sizeof(VertexId) * 2;
		} else {
			edge_unit = sizeof(VertexId) * 2 + sizeof(Weight);
		}

		memory_bytes = 1024l*1024l*1024l*1024l; // assume RAM capacity is very large
		partition_batch = partitions;
		vertex_data_bytes = 0;

		char filename[1024];
		fsize = new long * [partitions];
		for (int i=0;i<partitions;i++) {
			fsize[i] = new long [partitions];
			for (int j=0;j<partitions;j++) {
				sprintf(filename, "%s/block-%d-%d", path.c_str(), i, j);
				fsize[i][j] = file_size(filename);
			}
		}

		long bytes;

		column_offset = new long [partitions*partitions+1];
		int fin_column_offset = open((path+"/column_offset").c_str(), O_RDONLY);
		bytes = read(fin_column_offset, column_offset, sizeof(long)*(partitions*partitions+1));
		assert(bytes==static_cast<unsigned>(sizeof(long)*(partitions*partitions+1)));
		close(fin_column_offset);

		row_offset = new long [partitions*partitions+1];
		int fin_row_offset = open((path+"/row_offset").c_str(), O_RDONLY);
		bytes = read(fin_row_offset, row_offset, sizeof(long)*(partitions*partitions+1));
		assert(bytes==static_cast<unsigned>(sizeof(long)*(partitions*partitions+1)));
		close(fin_row_offset);
		if(c==-500) return;
	}

	Bitmap * alloc_bitmap() {
		return new Bitmap(vertices);
	}

	template <typename T>
	T stream_vertices(std::function<T(VertexId)> process, Bitmap * bitmap = nullptr, T zero = 0,
		std::function<void(std::pair<VertexId,VertexId>)> pre = f_none_1,
		std::function<void(std::pair<VertexId,VertexId>)> post = f_none_1) {
		T value = zero;
		if (bitmap==nullptr && vertex_data_bytes > (0.8 * memory_bytes)) {
			for (int cur_partition=0;cur_partition<partitions;cur_partition+=partition_batch) {
				VertexId begin_vid, end_vid;
				begin_vid = get_partition_range(vertices, partitions, cur_partition).first;
				if (cur_partition+partition_batch>=partitions) {
					end_vid = vertices;
				} else {
					end_vid = get_partition_range(vertices, partitions, cur_partition+partition_batch).first;
				}
				pre(std::make_pair(begin_vid, end_vid));
				#pragma omp parallel for schedule(dynamic) num_threads(parallelism)
				for (int partition_id=cur_partition;partition_id<cur_partition+partition_batch;partition_id++) {
					if (partition_id < partitions) {
						T local_value = zero;
						VertexId begin_vid, end_vid;
						std::tie(begin_vid, end_vid) = get_partition_range(vertices, partitions, partition_id);
						for (VertexId i=begin_vid;i<end_vid;i++) {
							local_value += process(i);
						}
						write_add(&value, local_value);
					}
				}
				#pragma omp barrier
				post(std::make_pair(begin_vid, end_vid));
			}
		} else {
			#pragma omp parallel for schedule(dynamic) num_threads(parallelism)
			for (int partition_id=0;partition_id<partitions;partition_id++) {
				T local_value = zero;
				VertexId begin_vid, end_vid;
				std::tie(begin_vid, end_vid) = get_partition_range(vertices, partitions, partition_id);
				if (bitmap==nullptr) {
					for (VertexId i=begin_vid;i<end_vid;i++) {
						local_value += process(i);
					}
				} else {
					VertexId i = begin_vid;
					while (i<end_vid) {
						unsigned long word = bitmap->data[WORD_OFFSET(i)];
						if (word==0) {
							i = (WORD_OFFSET(i) + 1) << 6;
							continue;
						}
						size_t j = BIT_OFFSET(i);
						word = word >> j;
						while (word!=0) {
							if (word & 1) {
								local_value += process(i);
							}
							i++;
							j++;
							word = word >> 1;
							if (i==end_vid) break;
						}
						i += (64 - j);
					}
				}
				write_add(&value, local_value);
			}
			#pragma omp barrier
		}
		return value;
	}

	void set_partition_batch(long bytes) {
		int x = (int)ceil(bytes / (0.8 * memory_bytes));
		partition_batch = partitions / x;
	}

	template <typename... Args>
	void hint(Args... args);

	template <typename A>
	void hint(BigVector<A> & a) {
		long bytes = sizeof(A) * a.length;
		set_partition_batch(bytes);
	}

	template <typename A, typename B>
	void hint(BigVector<A> & a, BigVector<B> & b) {
		long bytes = sizeof(A) * a.length + sizeof(B) * b.length;
		set_partition_batch(bytes);
	}

	template <typename A, typename B, typename C>
	void hint(BigVector<A> & a, BigVector<B> & b, BigVector<C> & c) {
		long bytes = sizeof(A) * a.length + sizeof(B) * b.length + sizeof(C) * c.length;
		set_partition_batch(bytes);
	}

	template <typename T>
	T stream_edges(T * parent_data, Bitmap * active_out, Bitmap * bitmap = nullptr, T zero = 0, int update_mode = 1,
		std::function<void(std::pair<VertexId,VertexId> vid_range)> pre_source_window = f_none_1,
		std::function<void(std::pair<VertexId,VertexId> vid_range)> post_source_window = f_none_1,
		std::function<void(std::pair<VertexId,VertexId> vid_range)> pre_target_window = f_none_1,
		std::function<void(std::pair<VertexId,VertexId> vid_range)> post_target_window = f_none_1) {
		if (bitmap==nullptr) {
			for (int i=0;i<partitions;i++) {
				should_access_shard[i] = true;
			}
		} else {
			for (int i=0;i<partitions;i++) {
				should_access_shard[i] = false;
			}
			#pragma omp parallel for schedule(dynamic) num_threads(parallelism)
			for (int partition_id=0;partition_id<partitions;partition_id++) {
				VertexId begin_vid, end_vid;
				std::tie(begin_vid, end_vid) = get_partition_range(vertices, partitions, partition_id);
				VertexId i = begin_vid;
				while (i<end_vid) {
					unsigned long long int word = bitmap->data[WORD_OFFSET(i)];
					if (word!=0) {
						should_access_shard[partition_id] = true;
						break;
					}
					i = (WORD_OFFSET(i) + 1) << 6;
				}
			}
			#pragma omp barrier
		}

		T value = zero;
		Queue<std::tuple<int, long, long> > tasks(65536);
		std::vector<std::thread> threads;
		long read_bytes = 0;

		long total_bytes = 0;
		for (int i=0;i<partitions;i++) {
			if (!should_access_shard[i]) continue;
			for (int j=0;j<partitions;j++) {
				total_bytes += fsize[i][j];
			}
		}
		int read_mode;
		if (memory_bytes < total_bytes) {
			read_mode = O_RDONLY | O_SYNC;
			// printf("use direct I/O\n");
		} else {
			read_mode = O_RDONLY;
			// printf("use buffered I/O\n");
		}

		// bufferをmallocにより確保
		char *buffer_mem = (char*)malloc(sizeof(char)*IOSIZE);
		memset(buffer_mem, 0, IOSIZE);

		// GPUで計算した値を集計するための領域を確保
        T *local_value_h = (T*)calloc(sizeof(T), 1);
        T *local_value_d;
        CHECK(cudaMalloc((void**)&local_value_d, sizeof(T)*1));
		CHECK(cudaMemset(local_value_d, -1, sizeof(T)*1));

		// parentのデバイス側の領域を確保
		T *parent_data_d;
		CHECK(cudaMalloc((void**)&parent_data_d, sizeof(T)*vertices));
		CHECK(cudaMemcpy(parent_data_d, parent_data, sizeof(T)*vertices, cudaMemcpyHostToDevice));

		int active_size = WORD_OFFSET(vertices-1) + 1; // active_inとactive_outのサイズ

		// active_inのデバイス側の領域を確保
		unsigned long long int *active_in_d;
        CHECK(cudaMalloc((void**)&active_in_d, sizeof(unsigned long long int)*active_size));
        CHECK(cudaMemcpy(active_in_d, bitmap->data, sizeof(unsigned long long int)*active_size, cudaMemcpyHostToDevice));

		// active_outのデバイス側の領域を確保
		unsigned long long int *active_out_d;
        CHECK(cudaMalloc((void**)&active_out_d, sizeof(unsigned long long int)*active_size));
        CHECK(cudaMemcpy(active_out_d, active_out->data, sizeof(unsigned long long int)*active_size, cudaMemcpyHostToDevice));

		// エッジのホスト側領域確保
		int *edge_h = (int*)malloc(sizeof(int)*MAX_EDGES*2);
		memset(edge_h, -1, sizeof(int)*MAX_EDGES*2);

		// エッジのデバイス側領域確保
		int *edge_d;
		CHECK(cudaMalloc((void**)&edge_d, sizeof(int)*MAX_EDGES*2));

		int fin;
		long offset = 0;
		switch(update_mode) {
		case 0: // source oriented update
			// threads.clear();
			// for (int ti=0;ti<parallelism;ti++) {
			// 	threads.emplace_back([&](int thread_id){
			// 		T local_value = zero;
			// 		long local_read_bytes = 0;
			// 		while (true) {
			// 			int fin;
			// 			long offset, length;
			// 			std::tie(fin, offset, length) = tasks.pop();
			// 			if (fin==-1) break;
			// 			char * buffer = buffer_pool[thread_id];
			// 			long bytes = pread(fin, buffer, length, offset);
			// 			assert(bytes>0);
			// 			local_read_bytes += bytes;
			// 			// CHECK: start position should be offset % edge_unit
			// 			for (long pos=offset % edge_unit;pos+edge_unit<=bytes;pos+=edge_unit) {
			// 				VertexId & src = *(VertexId*)(buffer+pos);
			// 				VertexId & dst = *(VertexId*)(buffer+pos+sizeof(VertexId));
			// 				if (bitmap->data==nullptr || bitmap->data[WORD_OFFSET(src)] & (1ul<<BIT_OFFSET(src))) {
			// 					local_value += process(src, dst, parent_data, active_out->data, 0);
			// 				}
			// 			}
			// 		}
			// 		write_add(&value, local_value);
			// 		write_add(&read_bytes, local_read_bytes);
			// 	}, ti);
			// }
			// fin = open((path+"/row").c_str(), read_mode);
			// //posix_fadvise(fin, 0, 0, POSIX_FADV_SEQUENTIAL); //This is mostly useless on modern system
			// for (int i=0;i<partitions;i++) {
			// 	if (!should_access_shard[i]) continue;
			// 	for (int j=0;j<partitions;j++) {
			// 		long begin_offset = row_offset[i*partitions+j];
			// 		if (begin_offset - offset >= PAGESIZE) {
			// 			offset = begin_offset / PAGESIZE * PAGESIZE;
			// 		}
			// 		long end_offset = row_offset[i*partitions+j+1];
			// 		if (end_offset <= offset) continue;
			// 		while (end_offset - offset >= IOSIZE) {
			// 			tasks.push(std::make_tuple(fin, offset, IOSIZE));
			// 			offset += IOSIZE;
			// 		}
			// 		if (end_offset > offset) {
			// 			tasks.push(std::make_tuple(fin, offset, (end_offset - offset + PAGESIZE - 1) / PAGESIZE * PAGESIZE));
			// 			offset += (end_offset - offset + PAGESIZE - 1) / PAGESIZE * PAGESIZE;
			// 		}
			// 	}
			// }
			// for (int i=0;i<parallelism;i++) {
			// 	tasks.push(std::make_tuple(-1, 0, 0));
			// }
			// for (int i=0;i<parallelism;i++) {
			// 	threads[i].join();
			// }
			break;
		case 1: // target oriented update
			fin = open((path+"/column").c_str(), read_mode);
			//posix_fadvise(fin, 0, 0, POSIX_FADV_SEQUENTIAL); //This is mostly useless on modern system

			for (int cur_partition=0;cur_partition<partitions;cur_partition+=partition_batch) {
				VertexId begin_vid, end_vid;
				begin_vid = get_partition_range(vertices, partitions, cur_partition).first;
				if (cur_partition+partition_batch>=partitions) {
					end_vid = vertices;
				} else {
					end_vid = get_partition_range(vertices, partitions, cur_partition+partition_batch).first;
				}
				pre_source_window(std::make_pair(begin_vid, end_vid));
				// printf("pre %d %d\n", begin_vid, end_vid);

                offset = 0;
				int count = 0;
				for (int j=0;j<partitions;j++) {
					for (int i=cur_partition;i<cur_partition+partition_batch;i++) {
						if (i>=partitions) break;
						if (!should_access_shard[i]) continue;
						long begin_offset = column_offset[j*partitions+i];
						if (begin_offset - offset >= PAGESIZE) {
							offset = begin_offset / PAGESIZE * PAGESIZE;
						}
						long end_offset = column_offset[j*partitions+i+1];
						if (end_offset <= offset) continue;
						while (end_offset - offset >= IOSIZE) {
							tasks.push(std::make_tuple(fin, offset, IOSIZE));
							count++;
							offset += IOSIZE;
						}
						if (end_offset > offset) {
							tasks.push(std::make_tuple(fin, offset, (end_offset - offset + PAGESIZE - 1) / PAGESIZE * PAGESIZE));
							count++;
							offset += (end_offset - offset + PAGESIZE - 1) / PAGESIZE * PAGESIZE;
						}
					}
				}
				// printf("count=%d\n", count);
				
				// cudaStream_t streams[count];
				// for (int i = 0; i < count; i++) {
				// 	CHECK(cudaStreamCreate(&streams[i]));
				// }

                tasks.push(std::make_tuple(-1, 0, 0));

				T local_value = zero;
				CHECK(cudaMemset(local_value_d, 0, sizeof(T)*1));
				long local_read_bytes = 0;
				
				// printf("remain_size=%ld\n", 2l*1024l*1024l*1024l - sizeof(T)*vertices - sizeof(unsigned long long int)*active_size*2 - sizeof(T) - sizeof(int)*IOSIZE/edge_unit*2);
				// printf("edge_size=%ld\n", sizeof(int)*IOSIZE/edge_unit*2);

				int count_while = 0;
				int local_edges = 0;
				while (true) {
					int fin = -1;
					long offset, length;
					std::tie(fin, offset, length) = tasks.pop();
					if (fin==-1) break;
					char * buffer = buffer_mem;
					long bytes = pread(fin, buffer, length, offset);
					assert(bytes>0);
					local_read_bytes += bytes;

					int edges = (bytes - (offset % edge_unit)) / edge_unit; // 読み込まれたエッジ数

					if (local_edges + edges > MAX_EDGES) {
						cudaEvent_t time1, time2, time3;
						cudaEventCreate(&time1);
						cudaEventCreate(&time2);
						cudaEventCreate(&time3);
						
						cudaEventRecord(time1, 0);

						CHECK(cudaMemcpy(edge_d, edge_h, sizeof(int)*local_edges*2, cudaMemcpyHostToDevice));

						cudaEventRecord(time2, 0);
						cudaEventSynchronize(time2);

						process_e<<<(local_edges+BS-1)/BS, BS>>>(edge_d, parent_data_d, active_in_d, active_out_d, local_value_d, local_edges);

						cudaEventRecord(time3, 0);
						cudaEventSynchronize(time3);

						float time12;
						cudaEventElapsedTime(&time12, time1, time2);

						float time23;
						cudaEventElapsedTime(&time23, time2, time3);

						cudaEventDestroy(time1);
						cudaEventDestroy(time2);
						cudaEventDestroy(time3);

						printf("block %d:\nedges=%d\n", count_while, local_edges);
						printf("time12: %.2fms\n", time12);
						printf("time23: %.2fms\n\n", time23);
						local_edges = 0;
						count_while++;
					}

					// CHECK: start position should be offset % edge_unit

					// ホスト領域のソース頂点配列とデスティネーション頂点配列に読み込まれた値を格納
					long pos = offset % edge_unit;
					for (int i = 0; i < edges; i++) {
						VertexId & src = *(VertexId*)(buffer+pos);
						VertexId & dst = *(VertexId*)(buffer+pos+sizeof(VertexId));
						edge_h[(local_edges + i)*2] = src;
						edge_h[(local_edges + i)*2+1] = dst;
						pos += edge_unit;
						if (src < begin_vid || src >= end_vid) {
							continue;
						}
					}
					
					local_edges += edges;
					// // process(edge_h, parent_data, bitmap->data, active_out->data, local_value_h, edges);
				}

				if (local_edges != 0) {
					cudaEvent_t time1, time2, time3;
					cudaEventCreate(&time1);
					cudaEventCreate(&time2);
					cudaEventCreate(&time3);
					
					cudaEventRecord(time1, 0);

					CHECK(cudaMemcpy(edge_d, edge_h, sizeof(int)*local_edges*2, cudaMemcpyHostToDevice));

					cudaEventRecord(time2, 0);
					cudaEventSynchronize(time2);

					process_e<<<(local_edges+BS-1)/BS, BS>>>(edge_d, parent_data_d, active_in_d, active_out_d, local_value_d, local_edges);

					cudaEventRecord(time3, 0);
					cudaEventSynchronize(time3);

					float time12;
					cudaEventElapsedTime(&time12, time1, time2);

					float time23;
					cudaEventElapsedTime(&time23, time2, time3);

					cudaEventDestroy(time1);
					cudaEventDestroy(time2);
					cudaEventDestroy(time3);

					printf("block %d:\nedges=%d\n", count_while, local_edges);
					printf("time12: %.2fms\n", time12);
					printf("time23: %.2fms\n\n", time23);
				}

				// for (int i = 0; i < count; i++) {
				// 	CHECK(cudaStreamDestroy(streams[i]));
				// }
				CHECK(cudaMemcpy(local_value_h, local_value_d, sizeof(T)*1, cudaMemcpyDeviceToHost));
				local_value = *local_value_h;
				write_add(&value, local_value);
				write_add(&read_bytes, local_read_bytes);
				post_source_window(std::make_pair(begin_vid, end_vid));
			}

			// parentとactive_outのホスト側の領域にデバイス側の領域からコピー
			CHECK(cudaMemcpy(parent_data, parent_data_d, sizeof(T)*vertices, cudaMemcpyDeviceToHost));
			CHECK(cudaMemcpy(active_out->data, active_out_d, sizeof(unsigned long long int)*active_size, cudaMemcpyDeviceToHost));

			break;
		default:
			assert(false);
		}
		free(buffer_mem);
		free(local_value_h);
		CHECK(cudaFree(local_value_d));
		CHECK(cudaFree(parent_data_d));
		CHECK(cudaFree(active_in_d));
		CHECK(cudaFree(active_out_d));
		free(edge_h);
		CHECK(cudaFree(edge_d));

		close(fin);
		// printf("streamed %ld bytes of edges\n", read_bytes);
		return value;
	}
};

#endif
