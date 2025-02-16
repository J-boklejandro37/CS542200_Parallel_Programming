// Dynamic load balancing without Class -> 89.90 s

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <memory>
#include <png.h>
#include <pthread.h>
#include <sched.h>
#include <stdexcept>
#include <vector>

// Add these structures at the top of your code
struct Task {
    int row;  // The row to process
};

struct TaskQueue {
    std::vector<Task> tasks;
    pthread_mutex_t mutex;
    size_t current_task;
    bool finished;
    
    TaskQueue(int height) : current_task(0), finished(false) {
        pthread_mutex_init(&mutex, nullptr);
        // Create tasks for each row
        for (int i = 0; i < height; i++) {
            tasks.emplace_back(i);
        }
    }
    
    ~TaskQueue() {
        pthread_mutex_destroy(&mutex);
    }
};
TaskQueue* task_queue;

double left, right, lower, upper;
int width, height, iters, thread_num;
std::unique_ptr<int[]> image;

void write_png(const std::string& filename)
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
            int p = image[(height - 1 - y) * width + x];
            png_bytep color = row.get() + x * 3;
            if (p != iters)
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

void* compute_Mandelbrot(void* arg)
{
    double x_offset = (right - left) / width;
    double y_offset = (upper - lower) / height;

    // Keep the SIMD constants
    __m512d vec_two = _mm512_set1_pd(2.0);
    __m512d vec_four = _mm512_set1_pd(4.0);
    __m512d vec_x_offset = _mm512_set1_pd(x_offset);
    __m512d vec_left = _mm512_set1_pd(left);

    while (true) {
        // Get next task
        Task current_task;
        bool has_task = false;
        
        pthread_mutex_lock(&task_queue->mutex);
        if (task_queue->current_task < task_queue->tasks.size()) {
            current_task = task_queue->tasks[task_queue->current_task++];
            has_task = true;
        }
        pthread_mutex_unlock(&task_queue->mutex);
        
        if (!has_task) break;  // No more tasks

        // Process the row (j is now current_task.row)
        int j = current_task.row;
        double y0 = j * y_offset + lower;
        __m512d vec_y0 = _mm512_set1_pd(y0);

        // Rest of your existing row processing code remains the same
        int i;
        for (i = 0; i < width - 7; i += 8) {
            // Each x0 = i * x_offset + left
            __m512d vec_x0 = _mm512_fmadd_pd(
                _mm512_set_pd(i+7, i+6, i+5, i+4, i+3, i+2, i+1, i),
                vec_x_offset,
                vec_left
            );

            // Initialize vectors
            __m512d vec_x = _mm512_setzero_pd();
            __m512d vec_y = _mm512_setzero_pd();
            __m512d vec_x2 = _mm512_setzero_pd();
            __m512d vec_y2 = _mm512_setzero_pd();
            __m512d vec_length_squared = _mm512_setzero_pd();
            __m256i vec_repeats = _mm256_setzero_si256(); //  For storing back. *image is a 32-bit integer, meaning 256 / 32 = 8 integers 
            __mmask8 mask = 0xFF; // All 8 bits set
            
            for (int iter = 0; iter < iters; ++iter) {
                vec_x2 = _mm512_mul_pd(vec_x, vec_x);
                vec_y2 = _mm512_mul_pd(vec_y, vec_y);

                // Length_squared = x^2 + y^2
                vec_length_squared = _mm512_add_pd(vec_x2, vec_y2);
                
                // Check which elements are still within bound. => If length_sqared < 4: set mask bit = 1
                mask = _mm512_cmp_pd_mask(vec_length_squared, vec_four, _CMP_LT_OS);
                if (!mask) break;

                // Calculate 2xy
                __m512d vec_2xy = _mm512_mul_pd(_mm512_mul_pd(vec_x, vec_y), vec_two);

                // New x = x^2 - y^2 + x0
                vec_x = _mm512_add_pd(_mm512_sub_pd(vec_x2, vec_y2), vec_x0);

                // New y = 2xy + y0
                vec_y = _mm512_add_pd(vec_2xy, vec_y0);

                // Only increment repeats for elements still within bounds
                vec_repeats = _mm256_mask_add_epi32(
                    vec_repeats,            // src: values to keep if mask == 0
                    mask,                   // mask: elements to operate on
                    vec_repeats,            // a: first operand
                    _mm256_set1_epi32(1)    // b: second operand
                );
            }
            
            // Store directly to image array
            // Endianess is automatically addressed, so the rightmost will be put first
            // _mm256_storeu_si256((__m256i*)&image[j * width + i], vec_repeats); // Traditional method, needs type casting
            _mm256_storeu_epi32(&image[j * width + i], vec_repeats); // Newer method, don't need type casting
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

    return nullptr;
}

int main(int argc, char** argv) {
    /* detect how many CPUs are available */
    cpu_set_t cpu_set;
    sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
    thread_num = CPU_COUNT(&cpu_set);
    std::cout << thread_num << " cpus available\n";

    /* argument parsing */
    assert(argc == 9);
    const std::string filename = argv[1];
    iters = strtol(argv[2], 0, 10);
    left = strtod(argv[3], 0);
    right = strtod(argv[4], 0);
    lower = strtod(argv[5], 0);
    upper = strtod(argv[6], 0);
    width = strtol(argv[7], 0, 10);
    height = strtol(argv[8], 0, 10);

    /* allocate memory for image */
    image = std::make_unique<int[]>(width * height);
    assert(image);

    // Initialize task queue
    task_queue = new TaskQueue(height);

    pthread_t threads[thread_num];
    
    // Create threads
    for (int i = 0; i < thread_num; ++i) {
        pthread_create(&threads[i], nullptr, compute_Mandelbrot, nullptr);
    }

    // Wait for all threads
    for (int i = 0; i < thread_num; ++i) {
        pthread_join(threads[i], nullptr);
    }

    // Clean up task queue
    delete task_queue;

    /* draw and cleanup */
    write_png(filename);

    return 0;
}
