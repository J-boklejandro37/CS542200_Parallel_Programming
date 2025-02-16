CC = gcc
CXX = g++
CFLAGS = -lm -O3
lab2_pthread: CFLAGS += -pthread
lab2_omp: CFLAGS += -fopenmp
lab2_hybrid: CC = mpicc
lab2_hybrid: CXX = mpicxx
lab2_hybrid: CFLAGS += -fopenmp
CXXFLAGS = $(CFLAGS)
TARGETS = lab2_pthread lab2_omp lab2_hybrid

.PHONY: all clean vector1 vector2

all: $(TARGETS)

vector1: vectorize_example.cc
	g++ -march=native vectorize_example.cc -o vectorize_example
	./vectorize_example

vector2: vectorize_example.cc
	g++ -lm -O3 -mavx512f vectorize_example.cc -o vectorize_example
	./vectorize_example

clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
