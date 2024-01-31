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

#include "../core/graph_pr.cuh"
#include "../core/common.h"

#define GNUPLOT_PATH "/mnt/c/gnuplot/bin/gnuplot.exe"

int main(int argc, char ** argv) {
	if (argc<3) {
		fprintf(stderr, "usage: pagerank [path] [iterations] [memory budget in GB]\n");
		exit(-1);
	}
	std::string path = argv[1];
	int iterations = atoi(argv[2]);
	long memory_bytes = (argc>=4)?atol(argv[3])*1024l*1024l*1024l:8l*1024l*1024l*1024l;

	Graph graph(path);
	graph.set_memory_bytes(memory_bytes);
	BigVector<VertexId> degree(graph.path+"/degree", graph.vertices);
	BigVector<VertexId> in_edges(graph.path+"/in_edges", graph.vertices);
	BigVector<float> pagerank(graph.path+"/pagerank", graph.vertices);
	BigVector<float> sum(graph.path+"/sum", graph.vertices);

	long vertex_data_bytes = (long)graph.vertices * ( sizeof(VertexId) + sizeof(float) + sizeof(float) );
	graph.set_vertex_data_bytes(vertex_data_bytes);

	double begin_time = get_time();

	degree.fill(0);
	graph.stream_edges<VertexId>(
		[&](Edge & e){
			write_add(&degree[e.source], 1);
			return 0;
		}, nullptr, 0, 0
	);
	printf("degree calculation used %.2f seconds\n", get_time() - begin_time);

	in_edges.fill(0);
	graph.stream_edges<VertexId>(
		[&](Edge & e){
			write_add(&in_edges[e.target], 1);
			return 0;
		}, nullptr, 0, 0
	);
	int index;
	int max = 0;
	for (int i = 0; i < graph.vertices; i++) {
		if (max < in_edges[i]) {
			max = in_edges[i];
			index = i;
		}
	}
	printf("max=%d\n", max);
	printf("index=%d\n", index);

	BigVector<VertexId> in_edges_num(graph.path+"/in_edges_num", max+1);
	in_edges_num.fill(0);

	for (int i = 0; i < graph.vertices; i++) {
		in_edges_num[in_edges[i]]++;
	}
	
	VertexId *degree_d;
	CHECK(cudaMalloc((void**)&degree_d, sizeof(VertexId)*graph.vertices));
	CHECK(cudaMemcpy(degree_d, degree.data, sizeof(VertexId)*graph.vertices, cudaMemcpyHostToDevice));

	pagerank.fill(0);

	float *pagerank_d;
	CHECK(cudaMalloc((void**)&pagerank_d, sizeof(float)*graph.vertices));
	CHECK(cudaMemset(pagerank_d, 0, sizeof(float)*graph.vertices));

	float *sum_d;
	CHECK(cudaMalloc((void**)&sum_d, sizeof(float)*graph.vertices));

	for (int iter=0; iter < iterations; iter++) {
		sum.fill(-INFINITY);
		CHECK(cudaMemcpy(sum_d, sum.data, sizeof(float)*graph.vertices, cudaMemcpyHostToDevice));
		graph.stream_edges_gpu<VertexId>(degree_d, pagerank_d, sum_d);
		// CHECK(cudaMemcpy(sum.data, sum_d, sizeof(float)*graph.vertices, cudaMemcpyDeviceToHost));
		// for (int i = 0; i < 10; i++) printf("sum[%d]=%f\n", i, sum[i]);
		graph.stream_vertices_gpu<VertexId>(pagerank_d, sum_d);
		// CHECK(cudaMemcpy(pagerank.data, pagerank_d, sizeof(float)*graph.vertices, cudaMemcpyDeviceToHost));
		// for (int i = 0; i < 10; i++) printf("pagerank[%d]=%f\n", i, pagerank[i]);
	}

	CHECK(cudaMemcpy(pagerank.data, pagerank_d, sizeof(float)*graph.vertices, cudaMemcpyDeviceToHost));
	for (int i = 0; i < 10; i++) printf("pagerank[%d]=%.2f\n", i, pagerank[i]);
}