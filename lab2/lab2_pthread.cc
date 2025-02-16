#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <cmath>

/*----------pthread usage tutorial----------*/
/*
int pthread_create(pthread_t* restrict <thread>,			// <thread> is a thread pointer (unique identifier for thread)
				   const pthread_attr_t* restrict <attr>,	// <attr> is thread attribute, usually nullptr
				   void* (*<routime>)(void*),				// <routine> is a function pointer
				   void* restrict <arg>						// <arg> is a pointer to arg, usually pass in "rank", so (void*)&tids[i]
				   );
int pthread_join(pthread_t <thread>,						// <thread> is a pthread
				 void** <retval>							// <retval> is a pointer to pointer, cuz it needs to retrieve a pointer
				 );
*/

int ncpus; // meaning "size" in mpi

unsigned long long radius, k, r_squared;

void* calc(void* thread_id)
{
	int tid = *static_cast<int*>(thread_id);

	unsigned long long count = 0;
    for (unsigned long long x = tid; x < radius; x += ncpus)
	{
        count += std::ceil(sqrtl(r_squared - x * x));
    }
	count %= k;

	pthread_exit(static_cast<void*>(new unsigned long long(count))); // need to pass it by the heap
}


int main(int argc, char* argv[]) {
	/*------------------store argv------------------*/
	assert(argc == 3);

	radius = std::stoull(argv[1]);
	k = std::stoull(argv[2]);
	r_squared = radius * radius;
	
	/*------------------initialize cpus------------------*/
	cpu_set_t cpuset;
	sched_getaffinity(0, sizeof(cpuset), &cpuset);
	ncpus = CPU_COUNT(&cpuset);

	/*------------------threading------------------*/
	pthread_t threads[ncpus]; // it's legal cuz pthread.h is a c library, meaning it can use Variable Length Arrays
	int* tids = new int[ncpus]; // cuz int tids[ncpus]; is illegal since ncpus is a runtime variable
	for (int idx = 0; idx < ncpus; idx += 1)
	{
		tids[idx] = idx;
		pthread_create(&threads[idx], nullptr, calc, (void*)&tids[idx]);
	}

	/*------------------joining------------------*/
	unsigned long long total_count = 0;
	for (int idx = 0; idx < ncpus; idx += 1)
	{
		void* thread_result;
		pthread_join(threads[idx], &thread_result);
		total_count += *static_cast<unsigned long long*>(thread_result);
		delete static_cast<unsigned long long*>(thread_result);
	}
	std::cout << (4 * total_count) % k << std::endl;
	
	delete[] tids;
	return 0;
}
