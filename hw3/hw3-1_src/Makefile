CC = gcc
CXX = g++ 
LDLIBS = -pthread -lm
CFLAGS = -O3 -fopenmp -march=native
CXXFLAGS = -O3 -fopenmp -std=c++17 -march=native
TARGETS = hw3-1

.PHONY: all
all: $(TARGETS)

hw3-1: hw3-1.cc
	$(CXX) $(CFLAGS) $< $(LDLIBS) -o $@

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
