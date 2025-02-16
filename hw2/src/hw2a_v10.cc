// Dynamic load balancing with Class, improve parallelism
// Double the parallelism -> 71.16 s
// Triple the parallelism -> 65.24 s
// Quadruple the parallelism -> 60.33 s

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <memory>
#include <pthread.h>
#include <png.h>
#include <sched.h>
#include <stdexcept>
#include <vector>

class TaskQueue
{
private:
    std::vector<int> rows;  // Store row numbers to process
    pthread_mutex_t mutex;
    size_t current_row;     // Iterator for current row

public:
    TaskQueue(int height) : current_row(0)
    {
        pthread_mutex_init(&mutex, nullptr);
        // Create tasks for each row
        rows.resize(height);
        int idx = 0;
        for (auto& x : rows) x = idx++;
    }

    ~TaskQueue()
    {
        pthread_mutex_destroy(&mutex);
    }

    // Get next row to process. Returns -1 if no more rows.
    int getNextRow()
    {
        pthread_mutex_lock(&mutex);
        int row = -1;
        if (current_row < rows.size())
        {
            row = rows[current_row++];
        }
        pthread_mutex_unlock(&mutex);
        return row;
    }
};

class PNGWriter
{
private:
    std::string filename;
    int width;
    int height;
    int iterations;
    const int* buffer;
    
    void validateInputs() const
    {
        if (!buffer) throw std::runtime_error("Invalid buffer");
        if (width <= 0 || height <= 0) throw std::runtime_error("Invalid dimensions");
    }

public:
    PNGWriter(const std::string& fname, int iters, int w, int h, const int* buf)
        : filename(fname), width(w), height(h), iterations(iters), buffer(buf) 
    {
        validateInputs();
    }

    void write() const 
    {
        std::unique_ptr<FILE, decltype(&fclose)> fp(fopen(filename.c_str(), "wb"), fclose);
        if (!fp) throw std::runtime_error("Failed to open file: " + filename);

        png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, nullptr, nullptr, nullptr);
        if (!png_ptr) throw std::runtime_error("Failed to create PNG write struct");

        png_infop info_ptr = png_create_info_struct(png_ptr);
        if (!info_ptr)
        {
            png_destroy_write_struct(&png_ptr, nullptr);
            throw std::runtime_error("Failed to create PNG info struct");
        }

        png_init_io(png_ptr, fp.get());
        png_set_IHDR(png_ptr, info_ptr, width, height, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                     PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
        png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
        png_write_info(png_ptr, info_ptr);
        png_set_compression_level(png_ptr, 1);

        std::unique_ptr<png_byte[]> row = std::make_unique<png_byte[]>(3 * width);
        const size_t row_size = 3 * width * sizeof(png_byte);

        for (int y = 0; y < height; ++y)
        {
            std::memset(row.get(), 0, row_size);
            for (int x = 0; x < width; ++x)
            {
                int p = buffer[(height - 1 - y) * width + x];
                png_bytep color = row.get() + x * 3;
                if (p != iterations)
                {
                    if (p & 16)
                    {
                        color[0] = 240;
                        color[1] = color[2] = (p & 15) << 4;
                    }
                    else color[0] = (p & 15) << 4;
                }
            }
            png_write_row(png_ptr, row.get());
        }

        png_write_end(png_ptr, nullptr);
        png_destroy_write_struct(&png_ptr, &info_ptr);
    }
};

class MandelbrotGenerator
{
private:
    double left, right, lower, upper;
    int width, height, iters, thread_num;
    std::unique_ptr<int[]> image;
    std::unique_ptr<TaskQueue> task_queue;

    struct ThreadData
    {
        MandelbrotGenerator* obj;
        int thread_id;
    };

    // need to use "static" so that the type is actually void* (*)(void*), or it will be void* (MandelbrotGenerator::*)(void*), meaning a "member function pointer"
    static void* wrapper(void* arg)
    {
        ThreadData* data = static_cast<ThreadData*>(arg);
        data->obj->compute(data->thread_id);
        return nullptr;
    }

    void compute(int thread_id)
    {
        double x_offset = (right - left) / width;
        double y_offset = (upper - lower) / height;

        // Constants for vectorized computation
        __m512d vec_two = _mm512_set1_pd(2.0);
        __m512d vec_four = _mm512_set1_pd(4.0);
        __m512d vec_x_offset = _mm512_set1_pd(x_offset);
        __m512d vec_left = _mm512_set1_pd(left);

        // Process rows dynamically
        while (true)
        {
            int j = task_queue->getNextRow();
            if (j == -1) break; // No more rows to process

            double y0 = j * y_offset + lower;
            __m512d vec_y0 = _mm512_set1_pd(y0);

            int i;
            for (i = 0; i < width - 31; i += 32)
            {
                // First set of 8 values
                __m512d vec_x0_1 = _mm512_fmadd_pd(
                    _mm512_set_pd(i+7, i+6, i+5, i+4, i+3, i+2, i+1, i),
                    vec_x_offset,
                    vec_left
                );
                
                // Second set of 8 values
                __m512d vec_x0_2 = _mm512_fmadd_pd(
                    _mm512_set_pd(i+15, i+14, i+13, i+12, i+11, i+10, i+9, i+8),
                    vec_x_offset,
                    vec_left
                );

                // Third set of 8 values
                __m512d vec_x0_3 = _mm512_fmadd_pd(
                    _mm512_set_pd(i+23, i+22, i+21, i+20, i+19, i+18, i+17, i+16),
                    vec_x_offset,
                    vec_left
                );

                // Fourth set of 8 values
                __m512d vec_x0_4 = _mm512_fmadd_pd(
                    _mm512_set_pd(i+31, i+30, i+29, i+28, i+27, i+26, i+25, i+24),
                    vec_x_offset,
                    vec_left
                );

                // Initialize vectors for all sets
                __m512d vec_x_1 = _mm512_setzero_pd();
                __m512d vec_y_1 = _mm512_setzero_pd();
                __m512d vec_x2_1 = _mm512_setzero_pd();
                __m512d vec_y2_1 = _mm512_setzero_pd();
                __m512d vec_length_squared_1 = _mm512_setzero_pd();
                
                __m512d vec_x_2 = _mm512_setzero_pd();
                __m512d vec_y_2 = _mm512_setzero_pd();
                __m512d vec_x2_2 = _mm512_setzero_pd();
                __m512d vec_y2_2 = _mm512_setzero_pd();
                __m512d vec_length_squared_2 = _mm512_setzero_pd();

                __m512d vec_x_3 = _mm512_setzero_pd();
                __m512d vec_y_3 = _mm512_setzero_pd();
                __m512d vec_x2_3 = _mm512_setzero_pd();
                __m512d vec_y2_3 = _mm512_setzero_pd();
                __m512d vec_length_squared_3 = _mm512_setzero_pd();

                __m512d vec_x_4 = _mm512_setzero_pd();
                __m512d vec_y_4 = _mm512_setzero_pd();
                __m512d vec_x2_4 = _mm512_setzero_pd();
                __m512d vec_y2_4 = _mm512_setzero_pd();
                __m512d vec_length_squared_4 = _mm512_setzero_pd();
                
                __m256i vec_repeats_1 = _mm256_setzero_si256();
                __m256i vec_repeats_2 = _mm256_setzero_si256();
                __m256i vec_repeats_3 = _mm256_setzero_si256();
                __m256i vec_repeats_4 = _mm256_setzero_si256();
                
                __mmask8 mask_1 = 0xFF;
                __mmask8 mask_2 = 0xFF;
                __mmask8 mask_3 = 0xFF;
                __mmask8 mask_4 = 0xFF;
                
                // Main iteration loop
                for (int iter = 0; iter < iters; ++iter) {
                    // Process first set
                    vec_x2_1 = _mm512_mul_pd(vec_x_1, vec_x_1);
                    vec_y2_1 = _mm512_mul_pd(vec_y_1, vec_y_1);
                    vec_length_squared_1 = _mm512_add_pd(vec_x2_1, vec_y2_1);
                    mask_1 = _mm512_cmp_pd_mask(vec_length_squared_1, vec_four, _CMP_LT_OS);

                    // Process second set
                    vec_x2_2 = _mm512_mul_pd(vec_x_2, vec_x_2);
                    vec_y2_2 = _mm512_mul_pd(vec_y_2, vec_y_2);
                    vec_length_squared_2 = _mm512_add_pd(vec_x2_2, vec_y2_2);
                    mask_2 = _mm512_cmp_pd_mask(vec_length_squared_2, vec_four, _CMP_LT_OS);

                    // Process third set
                    vec_x2_3 = _mm512_mul_pd(vec_x_3, vec_x_3);
                    vec_y2_3 = _mm512_mul_pd(vec_y_3, vec_y_3);
                    vec_length_squared_3 = _mm512_add_pd(vec_x2_3, vec_y2_3);
                    mask_3 = _mm512_cmp_pd_mask(vec_length_squared_3, vec_four, _CMP_LT_OS);

                    // Process fourth set
                    vec_x2_4 = _mm512_mul_pd(vec_x_4, vec_x_4);
                    vec_y2_4 = _mm512_mul_pd(vec_y_4, vec_y_4);
                    vec_length_squared_4 = _mm512_add_pd(vec_x2_4, vec_y2_4);
                    mask_4 = _mm512_cmp_pd_mask(vec_length_squared_4, vec_four, _CMP_LT_OS);

                    // Early exit if all masks are 0
                    if (!(mask_1 | mask_2 | mask_3 | mask_4)) break;

                    // Calculate 2xy for all sets
                    __m512d vec_2xy_1 = _mm512_mul_pd(_mm512_mul_pd(vec_x_1, vec_y_1), vec_two);
                    __m512d vec_2xy_2 = _mm512_mul_pd(_mm512_mul_pd(vec_x_2, vec_y_2), vec_two);
                    __m512d vec_2xy_3 = _mm512_mul_pd(_mm512_mul_pd(vec_x_3, vec_y_3), vec_two);
                    __m512d vec_2xy_4 = _mm512_mul_pd(_mm512_mul_pd(vec_x_4, vec_y_4), vec_two);

                    // Update x and y for first set
                    vec_x_1 = _mm512_add_pd(_mm512_sub_pd(vec_x2_1, vec_y2_1), vec_x0_1);
                    vec_y_1 = _mm512_add_pd(vec_2xy_1, vec_y0);

                    // Update x and y for second set
                    vec_x_2 = _mm512_add_pd(_mm512_sub_pd(vec_x2_2, vec_y2_2), vec_x0_2);
                    vec_y_2 = _mm512_add_pd(vec_2xy_2, vec_y0);

                    // Update x and y for third set
                    vec_x_3 = _mm512_add_pd(_mm512_sub_pd(vec_x2_3, vec_y2_3), vec_x0_3);
                    vec_y_3 = _mm512_add_pd(vec_2xy_3, vec_y0);

                    // Update x and y for fourth set
                    vec_x_4 = _mm512_add_pd(_mm512_sub_pd(vec_x2_4, vec_y2_4), vec_x0_4);
                    vec_y_4 = _mm512_add_pd(vec_2xy_4, vec_y0);

                    // Increment repeat counters for first set
                    vec_repeats_1 = _mm256_mask_add_epi32(
                        vec_repeats_1,
                        mask_1,
                        vec_repeats_1,
                        _mm256_set1_epi32(1)
                    );
                    
                    // Increment repeat counters for second set
                    vec_repeats_2 = _mm256_mask_add_epi32(
                        vec_repeats_2,
                        mask_2,
                        vec_repeats_2,
                        _mm256_set1_epi32(1)
                    );

                    // Increment repeat counters for third set
                    vec_repeats_3 = _mm256_mask_add_epi32(
                        vec_repeats_3,
                        mask_3,
                        vec_repeats_3,
                        _mm256_set1_epi32(1)
                    );

                    // Increment repeat counters for fourth set
                    vec_repeats_4 = _mm256_mask_add_epi32(
                        vec_repeats_4,
                        mask_4,
                        vec_repeats_4,
                        _mm256_set1_epi32(1)
                    );
                }

                // Store results for all sets
                _mm256_storeu_epi32(&image[j * width + i], vec_repeats_1);
                _mm256_storeu_epi32(&image[j * width + i + 8], vec_repeats_2);
                _mm256_storeu_epi32(&image[j * width + i + 16], vec_repeats_3);
                _mm256_storeu_epi32(&image[j * width + i + 24], vec_repeats_4);
            }
            // Finish the remaining
            for (; i < width; ++i)
            {
                double x0 = i * x_offset + left;
				double x = 0;
				double y = 0;
				double length_squared = 0;
				int repeats = 0;
				while (repeats < iters && length_squared < 4)
				{
					double temp = x * x - y * y + x0;
					y = 2 * x * y + y0;
					x = temp;
					length_squared = x * x + y * y;
					++repeats;
				}
				image[j * width + i] = repeats;
            }
        }
    }

public:
    MandelbrotGenerator(double l, double r, double low, double up, int w, int h, int iters, int thread_num)
        : left(l), right(r), lower(low), upper(up), width(w), height(h), iters(iters), thread_num(thread_num)
    {
        image = std::make_unique<int[]>(width * height);
        task_queue = std::make_unique<TaskQueue>(height); // TaskQueue is a Class, which takes in height as parameter in contructor
    }

    void generate()
    {
        pthread_t threads[thread_num]; // for thread instances
        ThreadData* thread_data = new ThreadData[thread_num]; // for callback function's argument

        // create threads and compute
        for (int i = 0; i < thread_num; ++i)
        {
            thread_data[i] = {this, i}; // pass in "this" so that the static function "wrapper" can access the member function "compute"
            pthread_create(&threads[i], nullptr, wrapper, &thread_data[i]); // can only pass in static function as callback function
        }

        // wait for all
        for (int i = 0; i < thread_num; ++i)
        {
            pthread_join(threads[i], nullptr);
        }
    }

    void saveToPNG(const std::string& filename) const
    {
        PNGWriter writer(filename, iters, width, height, image.get());
        writer.write();
    }
};

int main(int argc, char** argv)
{
    try
    {   // Detect available CPUs
        cpu_set_t cpu_set;
        sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
        int thread_num = CPU_COUNT(&cpu_set);
        std::cout << thread_num << " cpus available\n";

        // Validate arguments
        if (argc != 9) throw std::runtime_error("Invalid number of arguments");

        // Parse arguments
        const std::string filename = argv[1];
        int iterations = std::stoi(argv[2]);
        double left = std::stod(argv[3]);
        double right = std::stod(argv[4]);
        double lower = std::stod(argv[5]);
        double upper = std::stod(argv[6]);
        int width = std::stoi(argv[7]);
        int height = std::stoi(argv[8]);

        // Generate and save Mandelbrot set
        MandelbrotGenerator mandelbrot(left, right, lower, upper, width, height, iterations, thread_num);
        mandelbrot.generate();
        mandelbrot.saveToPNG(filename);

        return 0;
    } 
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}