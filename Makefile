
ROOT_DIR= $(shell pwd)
TARGETS= bin/firstPreprocessing bin/preprocess bin/bfs bin/wcc bin/pagerank bin/spmv bin/mis bin/radii bin/bfs_gpu

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

bin/bfs_gpu: examples/bfs.cu core/graph1.cuh
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(SYSLIBS)

clean:
	rm -rf $(TARGETS)