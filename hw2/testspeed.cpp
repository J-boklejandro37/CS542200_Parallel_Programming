#include <pthread.h>
#include <thread>
#include <chrono>
#include <iostream>
#include <vector>

const int NUM_THREADS = 4;
const int ITERATIONS = 10000000;

// Work function for both thread types
void* pthread_work(void* arg) {
    long sum = 0;
    for(int i = 0; i < ITERATIONS; i++) {
        sum += i;
    }
    return nullptr;
}

void thread_work() {
    long sum = 0;
    for(int i = 0; i < ITERATIONS; i++) {
        sum += i;
    }
}

// Benchmark functions
double benchmark_pthreads() {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<pthread_t> threads(NUM_THREADS);
    
    // Create threads
    for(int i = 0; i < NUM_THREADS; i++) {
        pthread_create(&threads[i], nullptr, pthread_work, nullptr);
    }
    
    // Join threads
    for(int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], nullptr);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

double benchmark_cpp_threads() {
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<std::thread> threads;
    threads.reserve(NUM_THREADS);
    
    // Create threads
    for(int i = 0; i < NUM_THREADS; i++) {
        threads.emplace_back(thread_work);
    }
    
    // Join threads
    for(auto& thread : threads) {
        thread.join();
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    return diff.count();
}

int main() {
    const int NUM_RUNS = 5;
    double pthread_total = 0, cpp_thread_total = 0;
    
    std::cout << "Running benchmarks..." << std::endl;
    
    for(int i = 0; i < NUM_RUNS; i++) {
        pthread_total += benchmark_pthreads();
        cpp_thread_total += benchmark_cpp_threads();
    }
    
    double pthread_avg = pthread_total / NUM_RUNS;
    double cpp_thread_avg = cpp_thread_total / NUM_RUNS;
    
    std::cout << "\nResults (average over " << NUM_RUNS << " runs):" << std::endl;
    std::cout << "pthread time: " << pthread_avg << " seconds" << std::endl;
    std::cout << "C++ thread time: " << cpp_thread_avg << " seconds" << std::endl;
    std::cout << "Difference: " << ((cpp_thread_avg / pthread_avg - 1) * 100) << "%" << std::endl;
    
    return 0;
}