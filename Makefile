
ROOT_DIR= $(shell pwd)
TARGETS= bin/firstPreprocessing bin/preprocess bin/bfs bin/wcc bin/pagerank bin/spmv bin/mis bin/radii

CXX?= g++
CXXFLAGS?= -O3 -Wall -std=c++11 -g -fopenmp -I$(ROOT_DIR)
NVCC=nvcc
NVCCFLAGS= -arch=sm_75 -O3 -std=c++11 -g -Xcompiler -fopenmp -I$(ROOT_DIR)
HEADERS= $(shell find . -name '*.hpp')

all: $(TARGETS)

bin/firstPreprocessing: tools/firstPreprocessing.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/preprocess: tools/preprocess.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) -o $@ $< $(SYSLIBS)

bin/bfs: examples/bfs.cpp $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(SYSLIBS)

bin/wcc: examples/wcc.cpp $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(SYSLIBS)

bin/pagerank: examples/pagerank.cpp $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(SYSLIBS)

bin/spmv: examples/spmv.cpp $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(SYSLIBS)

bin/mis: examples/mis.cpp $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(SYSLIBS)

bin/radii: examples/radii.cpp $(HEADERS)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(SYSLIBS)

clean:
	rm -rf $(TARGETS)

