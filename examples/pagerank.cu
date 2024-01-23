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
	BigVector<VertexId> degree(graph.path+"/degree", graph.vertices);
	BigVector<float> pagerank(graph.path+"/pagerank", graph.vertices);
	BigVector<float> new_pagerank(graph.path+"/new_pagerank", graph.vertices);

	long vertex_data_bytes = (long)graph.vertices * ( sizeof(VertexId) + sizeof(float) + sizeof(float) );
	graph.set_vertex_data_bytes(vertex_data_bytes);

	degree.fill(0);

	VertexId *degree_d;
	CHECK(cudaMalloc((void**)&degree_d, sizeof(VertexId)*graph.vertices));
	CHECK(cudaMemcpy(degree_d, degree.data, sizeof(VertexId)*graph.vertices, cudaMemcpyHostToDevice));

	// float *test = (float*)malloc(sizeof(float)*2);
	// test[0] = 0;
	// test[1] = 0;
	// float *test_d;
	// cudaMalloc((void**)&test_d, sizeof(float)*2);
	// cudaMemcpy(test_d, test, sizeof(float)*2, cudaMemcpyHostToDevice);
	// testLogAdd<<<1, 1024>>>(test_d);
	// cudaDeviceSynchronize();
	// cudaMemcpy(test, test_d, sizeof(float)*2, cudaMemcpyDeviceToHost);
	// for (int i = 0; i < 2; i++) {
	// 	printf("test[%d]=%.4f\n", i, test[i]);
	// }
	return 0;
}
