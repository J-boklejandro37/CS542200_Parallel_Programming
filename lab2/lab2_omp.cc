#include <iostream>
#include <string>
#include <vector>
#include <cassert>
#include <cmath>
#include <omp.h>

int main(int argc, char* argv[]) {
	/*------------------store argv------------------*/
	assert(argc == 3);
	unsigned long long radius = std::stoull(argv[1]);
	unsigned long long k = std::stoull(argv[2]);
	unsigned long long r_squared = radius * radius;

    /*------------------parallel calculation------------------*/
	unsigned long long total_count = 0;
	int num_threads;

    #pragma omp parallel reduction(+:total_count) // by default it's default(shared), meaning num_threads will be shared among threads
    {
		#pragma omp single
		{
			num_threads = omp_get_num_threads(); // needs to be done inside parallel region, otherwise will always be 1 (the master thread)
		}
	
        unsigned long long count = 0;
        for (unsigned long long x = omp_get_thread_num(); x < radius; x += num_threads)
        {
            count += std::ceil(sqrtl(r_squared - x * x));
        }
        total_count += count % k;
    }

    std::cout << (4 * total_count) % k << std::endl;
}
