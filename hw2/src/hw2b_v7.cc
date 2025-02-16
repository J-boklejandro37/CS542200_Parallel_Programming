// Same as version 5, remove MPI_Probe, remove work unit, use std::pair<int, int> instead -> (20241031) 99.19 s
// Triple the vectorization -> 81.19 s

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <memory> // std::unique_ptr
#include <mpi.h>
#include <omp.h>
#include <png.h>
#include <queue>
#include <sched.h>
#include <stdexcept>
#include <utility> // std::move
#include <vector>

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

        // Create a vector of rows for parallel processing
        std::vector<std::unique_ptr<png_byte[]>> all_rows;
        all_rows.reserve(height);
        for (int i = 0; i < height; ++i)
            all_rows.emplace_back(std::make_unique<png_byte[]>(3 * width));

        const size_t row_size = 3 * width * sizeof(png_byte);

        #pragma omp parallel for schedule(static)
        for (int y = 0; y < height; ++y)
        {
            auto& row = all_rows[y];
            std::memset(row.get(), 0, row_size);
            int base = (height - 1 - y) * width;
            for (int x = 0; x < width; ++x)
            {
                int p = buffer[base + x];
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
        }
        for (int y = 0; y < height; ++y)
        {
            png_write_row(png_ptr, all_rows[y].get());
        }

        png_write_end(png_ptr, nullptr);
        png_destroy_write_struct(&png_ptr, &info_ptr);
    }
};

class Master
{
private:
    double left, right, lower, upper;
    int width, height, iters;
    std::unique_ptr<int[]> image;

public:
    Master(double l, double r, double low, double up, int w, int h, int iters)
        : left(l), right(r), lower(low), upper(up), width(w), height(h), iters(iters)
    {
        image = std::make_unique<int[]>(width * height);
    }

    // Store the computed rows back to image
    void storeRows(int start_row, const std::vector<int>& buffer, int num_rows)
    {
        std::memcpy(image.get() + start_row * width, buffer.data(), num_rows * width * sizeof(int));
    }

    void saveToPNG(const std::string& filename) const
    {
        PNGWriter writer(filename, iters, width, height, image.get());
        writer.write();
    }
};

class Worker
{
private:
    double left, right, lower, upper;
    int width, height, iters;
    // Pre-computed constants for vectorization
    alignas(64) const double vec_init_sequence[8] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

public:
    Worker(double l, double r, double low, double up, int w, int h, int iters)
        : left(l), right(r), lower(low), upper(up), width(w), height(h), iters(iters)
    {
    }

    // For worker process
    std::vector<int> computeRows(int start_row, int num_rows) const
    {
        std::vector<int> buffer(width * num_rows);
        double x_offset = (right - left) / width;
        double y_offset = (upper - lower) / height;

        // Constants for vectorized computation
        // "set1_pd" is broadcasting single value, meaning it has less overhead
        const __m512d vec_two = _mm512_set1_pd(2.0);
        const __m512d vec_four = _mm512_set1_pd(4.0);
        const __m512d vec_x_offset = _mm512_set1_pd(x_offset);
        const __m512d vec_left = _mm512_set1_pd(left);
        // "set_pd" needs to assign value for each element, which has more overhead. Thus using pre-initialized sequence and loading into register is more efficient
        // _mm512_set_pd(i+7, i+6, i+5, i+4, i+3, i+2, i+1, i)
        const __m512d vec_sequence = _mm512_load_pd(vec_init_sequence);

        #pragma omp parallel for schedule(dynamic, 1)
        for (int j = 0; j < num_rows; ++j)
        {
            int global_j = start_row + j;
            double y0 = global_j * y_offset + lower;
            __m512d vec_y0 = _mm512_set1_pd(y0);

            int i;
            for (i = 0; i < width - 23; i += 24)
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
                
                __m256i vec_repeats_1 = _mm256_setzero_si256();
                __m256i vec_repeats_2 = _mm256_setzero_si256();
                __m256i vec_repeats_3 = _mm256_setzero_si256();
                
                __mmask8 mask_1 = 0xFF;
                __mmask8 mask_2 = 0xFF;
                __mmask8 mask_3 = 0xFF;
                
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

                    // Early exit if all masks are 0
                    if (!(mask_1 | mask_2 | mask_3)) break;

                    // Calculate 2xy for all sets
                    __m512d vec_2xy_1 = _mm512_mul_pd(_mm512_mul_pd(vec_x_1, vec_y_1), vec_two);
                    __m512d vec_2xy_2 = _mm512_mul_pd(_mm512_mul_pd(vec_x_2, vec_y_2), vec_two);
                    __m512d vec_2xy_3 = _mm512_mul_pd(_mm512_mul_pd(vec_x_3, vec_y_3), vec_two);

                    // Update x and y for first set
                    vec_x_1 = _mm512_add_pd(_mm512_sub_pd(vec_x2_1, vec_y2_1), vec_x0_1);
                    vec_y_1 = _mm512_add_pd(vec_2xy_1, vec_y0);

                    // Update x and y for second set
                    vec_x_2 = _mm512_add_pd(_mm512_sub_pd(vec_x2_2, vec_y2_2), vec_x0_2);
                    vec_y_2 = _mm512_add_pd(vec_2xy_2, vec_y0);

                    // Update x and y for third set
                    vec_x_3 = _mm512_add_pd(_mm512_sub_pd(vec_x2_3, vec_y2_3), vec_x0_3);
                    vec_y_3 = _mm512_add_pd(vec_2xy_3, vec_y0);

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
                }

                // Store results for all sets
                _mm256_storeu_epi32(&buffer[j * width + i], vec_repeats_1);
                _mm256_storeu_epi32(&buffer[j * width + i + 8], vec_repeats_2);
                _mm256_storeu_epi32(&buffer[j * width + i + 16], vec_repeats_3);
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
				buffer[j * width + i] = repeats;
            }
        }

        return std::move(buffer);
    }
};

void masterProcess(int num_processes, const std::string& filename, double left, double right, double lower, double upper, int width, int height, int iters);

void workerProcess(int rank, double left, double right, double lower, double upper, int width, int height, int iters);

int main(int argc, char** argv) {
    try
    {   // Initialize MPI
        int provided;
        MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Validate arguments
        if (argc != 9) {
            if (rank == 0) {
                throw std::runtime_error("Invalid number of arguments");
            }
            MPI_Finalize();
            return 1;
        }

        // Only master process (rank 0) should handle CPU detection
        if (rank == 0) {
            cpu_set_t cpu_set;
            sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
            std::cout << CPU_COUNT(&cpu_set) << " cpus available\n";
            std::cout << size << " MPI processes started\n";
        }

        // Parse arguments
        const std::string filename = argv[1];
        int iters = std::stoi(argv[2]);
        double left = std::stod(argv[3]);
        double right = std::stod(argv[4]);
        double lower = std::stod(argv[5]);
        double upper = std::stod(argv[6]);
        int width = std::stoi(argv[7]);
        int height = std::stoi(argv[8]);

        // master-worker process
        if (rank == 0) masterProcess(size, filename, left, right, lower, upper, width, height, iters);
        else workerProcess(rank, left, right, lower, upper, width, height, iters);
        
        MPI_Finalize();

        return 0;
    } 
    catch (const std::exception& e)
    {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        if (rank == 0) {
            std::cerr << "Error: " << e.what() << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
}

void masterProcess(int num_processes, const std::string& filename, double left, double right, double lower, double upper, int width, int height, int iters)
{
    // CHUNK_SIZE: num_rows for work unit
    const int CHUNK_SIZE = 60;

    Master master(left, right, lower, upper, width, height, iters);

    // Initialize work queue
    std::queue<std::pair<int, int>> work_queue;
    for (int row = 0; row < height; row += CHUNK_SIZE)
    {
        int num_rows = std::min(CHUNK_SIZE, height - row);
        work_queue.emplace(row, num_rows);
    }

    // Track finished workers
    int finished_workers = 0;

    // Waiting for work requests or results until all works are done
    // Need to make sure all processes are terminated
    MPI_Status status;
    while (!work_queue.empty() || finished_workers < num_processes - 1)
    {
        // Wait for worker's message
        std::pair<int, int> message{};
        MPI_Recv(&message.first, 2, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);

        int source = status.MPI_SOURCE;

        if (message.first == -1) // Meaning work request
        {
            // Dispatch the work if exists, or terminate the process
            if (!work_queue.empty())
            {
                auto unit = work_queue.front();
                work_queue.pop();
                MPI_Send(&unit.first, 2, MPI_INT, source, 0, MPI_COMM_WORLD);
            }
            else
            {   // Sent the terminate signal
                std::pair<int, int> mes{-1, 0};
                MPI_Send(&mes.first, 2, MPI_INT, source, 0, MPI_COMM_WORLD);
                finished_workers += 1;
            }
        }
        else // Meaning sending back result
        {
            // Receive the actual data and store it directly into image
            std::vector<int> result(message.second * width);
            MPI_Recv(result.data(), message.second * width, MPI_INT, source, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

            // Store onto image
            master.storeRows(message.first, result, message.second);
        }
    }

    // Save the final image
    master.saveToPNG(filename);
}

void workerProcess(int rank, double left, double right, double lower, double upper, int width, int height, int iters)
{
    Worker worker(left, right, lower, upper, width, height, iters);

    // Keep requesting for work until getting a termination, then break
    while (true)
    {
        // Request work from master
        std::pair<int, int> message{-1, 0};
        MPI_Send(&message.first, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
        
        // Receive header unit
        MPI_Recv(&message.first, 2, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        if (message.first == -1) break;

        // Do the work
        std::vector<int> result = worker.computeRows(message.first, message.second);
        
        // Sender the header back first
        MPI_Send(&message.first, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
        // Then send the computed data
        MPI_Send(result.data(), result.size(), MPI_INT, 0, 0, MPI_COMM_WORLD);
    }
}