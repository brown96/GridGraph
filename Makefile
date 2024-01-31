
ROOT_DIR= $(shell pwd)
TARGETS= bin/firstPreprocessing bin/preprocess bin/bfs bin/wcc bin/pagerank bin/spmv bin/mis bin/radii bin/bfs_gpu bin/bfs_cpu bin/pagerank_gpu bin/pagerank_cpu bin/incoming_edges

CXX?= g++
CXXFLAGS?= -O3 -Wall -std=c++11 -g -fopenmp -I$(ROOT_DIR)
NVCC=nvcc
NVCCFLAGS= -O3 -std=c++11 -arch compute_61 -code sm_61 -Xcompiler -fopenmp -I$(ROOT_DIR)
HEADERS= $(shell find . -name '*.hpp')

all: $(TARGETS)

bin/firstPreprocessing: tools/firstPreprocessing.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/preprocess: tools/preprocess.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/bfs: examples/bfs.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/wcc: examples/wcc.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/pagerank: examples/pagerank.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/spmv: examples/spmv.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/mis: examples/mis.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/radii: examples/radii.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/bfs_gpu: examples/bfs_rev.cu core/graph.cuh core/graph_bfs.cuh core/graph_cpu.cuh core/constants.hpp
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(SYSLIBS)

bin/bfs_cpu: examples/bfs.cu core/graph_cpu.cuh core/constants.hpp
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(SYSLIBS)

bin/pagerank_gpu: examples/pagerank.cu core/graph_pr.cuh core/constants.hpp
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(SYSLIBS)

bin/pagerank_cpu: examples/pagerank_cpu.cu core/graph_pr_cpu.cuh core/constants.hpp
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(SYSLIBS)

bin/incoming_edges: examples/incoming_edges.cu core/graph_pr.cuh core/constants.hpp
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(SYSLIBS)

clean:
	rm -rf $(TARGETS)