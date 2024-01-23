/*
Copyright (c) 2014-2015 Xiaowei Zhu, Tsinghua University

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

#include "../core/graph_bfs.cuh"
#include "../core/common.h"

int main(int argc, char ** argv) {
	if (argc<3) {
		fprintf(stderr, "usage: bfs [path] [start vertex id] [memory budget in GB]\n");
		exit(-1);
	}
	std::string path = argv[1];
	VertexId start_vid = atoi(argv[2]);
	long memory_bytes = (argc>=4)?atol(argv[3])*1024l*1024l*1024l:8l*1024l*1024l*1024l;

	Graph graph(path);
	graph.set_memory_bytes(memory_bytes);
	Bitmap * active_in = graph.alloc_bitmap();
	Bitmap * active_out = graph.alloc_bitmap();
	BigVector<VertexId> parent(graph.path+"/parent", graph.vertices);
	graph.set_vertex_data_bytes( graph.vertices * sizeof(VertexId) );

	active_out->clear();
	active_out->set_bit(start_vid);
	parent.fill(-1);
	parent[start_vid] = start_vid;
	VertexId active_vertices = 1;

	VertexId *parent_data_d;
	CHECK(cudaMalloc((void**)&parent_data_d, sizeof(VertexId)*graph.vertices));
	CHECK(cudaMemcpy(parent_data_d, parent.data, sizeof(VertexId)*graph.vertices, cudaMemcpyHostToDevice));

	unsigned long long int *active_in_d;
	CHECK(cudaMalloc((void**)&active_in_d, sizeof(unsigned long long int)*graph.active_size));
	CHECK(cudaMemcpy(active_in_d, active_in->data, sizeof(unsigned long long int)*graph.active_size, cudaMemcpyHostToDevice));

	unsigned long long int *active_out_d;
	CHECK(cudaMalloc((void**)&active_out_d, sizeof(unsigned long long int)*graph.active_size));
	CHECK(cudaMemcpy(active_out_d, active_out->data, sizeof(unsigned long long int)*graph.active_size, cudaMemcpyHostToDevice));

	double start_time = get_time();
	int iteration = 0;
	while (active_vertices!=0) {
		iteration++;
		printf("%7d: %d\n", iteration, active_vertices);
		CHECK(cudaMemcpy(active_in_d, active_out_d, sizeof(unsigned long long int)*graph.active_size, cudaMemcpyDeviceToDevice));
		CHECK(cudaMemset(active_out_d, 0, sizeof(unsigned long long int)*graph.active_size));
		graph.hint(parent);
		// double start_time_e = get_time();
		active_vertices = graph.stream_edges<VertexId>(parent_data_d, active_out_d, active_in_d);
		// double end_time_e = get_time();
		// printf("Total Stream: %.2fms\n\n", (end_time_e - start_time_e)*1000);
	}
	double end_time = get_time();

	CHECK(cudaMemcpy(parent.data, parent_data_d, sizeof(VertexId)*graph.vertices, cudaMemcpyDeviceToHost));

	int discovered_vertices = graph.stream_vertices<VertexId>([&](VertexId i){
		return parent[i]!=-1;
	});
	printf("discovered %d vertices from %d in %.2f seconds.\n", discovered_vertices, start_vid, end_time - start_time);

	return 0;
}
