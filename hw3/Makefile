NVFLAGS = -std=c++11 -O3 -Xptxas="-v" -arch=sm_61 -Xcompiler -fopenmp
HIPCCFLAGS = -std=c++11 -O3 --offload-arch=gfx90a

CXXFLAGS = -O3 -lm
LDFLAGS = -lm -lgomp
EXES = hw3-2 hw3-3

.PHONY: all clean test

all: $(EXES)

clean:
	rm -f $(EXES)

seq: seq.cc
	g++ $(CXXFLAGS) -o $@ $?

hw3-2: hw3-2.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-2-amd: hw3-2.hip
	hipcc $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<

hw3-3: hw3-3.cu
	nvcc $(NVFLAGS) $(LDFLAGS) -o $@ $?

hw3-3-amd: hw3-3.hip
	hipcc $(HIPCCFLAGS) $(LDFLAGS) -o $@ $<

testnv:
	srun -p nvidia -N1 -n1 -c2 --gres=gpu:1 nsys profile --trace=cuda,nvtx ./hw3-2 /share/testcases/hw3/p12k1 p12k1.out
testnv2:
	srun -p nvidia -N1 -n1 -c2 --gres=gpu:2 nsys profile --trace=cuda,nvtx ./hw3-3 /share/testcases/hw3/p12k1 p12k1.out

testamd:
	srun -p amd -N1 -n1 -c2 --gres=gpu:1 nsys profile --trace=cuda,nvtx ./hw3-2-amd /share/testcases/hw3/p12k1 p12k1.out
testamd2:
	srun -p amd -N1 -n1 -c2 --gres=gpu:2 nsys profile --trace=cuda,nvtx ./hw3-3-amd /share/testcases/hw3/p12k1 p12k1.out

