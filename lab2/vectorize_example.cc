#include <iostream>
#include <vector>
#include <immintrin.h>
#include <chrono>
#include <iomanip>

void multiple_and_add(float *a, float *b, float *c, float *d, int size){
    for(int i = 0; i < size; i++){
        a[i] = b[i] * c[i] + d[i];
    }
}

void vec_multiple_and_add(float *a, float *b, float *c, float *d, int size){

    int i;

    for(i = 0; i < size - 15; i += 16){

        // load data to special registers
        __m512 b_vec = _mm512_loadu_ps(&b[i]);
        __m512 c_vec = _mm512_loadu_ps(&c[i]);
        __m512 d_vec = _mm512_loadu_ps(&d[i]);

        // _mm512_fmadd_ps finish the multiplae and add operation
        // _mm512_storeu_ps store the result to a array
        _mm512_storeu_ps(&a[i], _mm512_fmadd_ps(b_vec, c_vec, d_vec));
    }

    // remaining elements
    for(; i < size; i++){
        a[i] = b[i] * c[i] + d[i];
    }
}

template<typename Func, typename... Args>
double measure_time(Func func, Args&&... args) {
    auto start = std::chrono::high_resolution_clock::now();
    func(std::forward<Args>(args)...);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

int main() {
    const int size = 10000000;  
    std::vector<float> a(size), b(size), c(size), d(size), a_vec(size);

    for(int i = 0; i < size; i++){
        b[i] = static_cast<float>(i % 100);  
        c[i] = static_cast<float>((i * 2) % 100);
        d[i] = static_cast<float>((i * 3) % 100);
    }

    double time_normal = measure_time(multiple_and_add, a.data(), b.data(), c.data(), d.data(), size);

    double time_vectorized = measure_time(vec_multiple_and_add, a_vec.data(), b.data(), c.data(), d.data(), size);

    bool results_match = std::equal(a.begin(), a.end(), a_vec.begin());

    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Results " << (results_match ? "match" : "do not match") << std::endl;
    std::cout << "Time taken by normal function: " << time_normal << " ms" << std::endl;
    std::cout << "Time taken by vectorized function: " << time_vectorized << " ms" << std::endl;
    std::cout << "Speedup: " << time_normal / time_vectorized << "x" << std::endl;

    return 0;
}