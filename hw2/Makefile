CC = gcc
CXX = g++
LDLIBS = -lpng
CFLAGS = -lm -O3
hw2a: CFLAGS += -pthread -march=native
hw2b: CC = mpicc
hw2b: CXX = mpicxx
hw2b: CFLAGS += -fopenmp -march=native
CXXFLAGS = $(CFLAGS)
TARGETS = hw2seq hw2a hw2b

.PHONY: all
all: $(TARGETS)

hw2a: hw2a.cc
	$(CXX) $(CXXFLAGS) $< $(LDLIBS) -o $@

hw2b: hw2b.cc
	$(CXX) $(CXXFLAGS) $< $(LDLIBS) -o $@

old_out.png: hw2seq_old
	srun -n1 -c1 ./$< $@ 10000 -2 2 -2 2 800 800

hw2seq: hw2seq.cc
	$(CXX) $(CXXFLAGS) $< $(LDLIBS) -o $@
	
new_out.png: hw2seq_new
	srun -n1 -c1 ./$< $@ 10000 -2.5 1.5 -2 2 800 800

hw2seq_new: hw2seq_new.cc
	$(CXX) $(CXXFLAGS) $< $(LDLIBS) -o $@

testspeed: testspeed.cpp
	$(CXX) $(CXXFLAGS) $< -pthread $(LDLIBS) -o $@
	./$@

.PHONY: clean
clean:
	rm -f $(TARGETS) $(TARGETS:=.o)
