// For profiling
// static load (continuous), vectorization -> Time: 340.47 s

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <memory>
#include <nvtx3/nvtx3.hpp>
#include <mpi.h>
#include <omp.h>
#include <png.h>
#include <sched.h>
#include <stdexcept>
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

        const size_t row_size = 3 * width * sizeof(png_byte);

        // Pre-compute all rows in parallel
        std::vector<std::vector<png_byte>> all_rows(height, std::vector<png_byte>(row_size));

        #pragma omp parallel for schedule(dynamic)
        for (int y = 0; y < height; ++y)
        {
            auto& row = all_rows[y];
            std::memset(row.data(), 0, row_size);
            for (int x = 0; x < width; ++x)
            {
                int p = buffer[(height - 1 - y) * width + x];
                png_bytep color = row.data() + x * 3;
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

        // Sequential write to file
        for (int y = 0; y < height; ++y)
        {
            png_write_row(png_ptr, all_rows[y].data());
        }

        png_write_end(png_ptr, nullptr);
        png_destroy_write_struct(&png_ptr, &info_ptr);
    }
};

class MandelbrotGenerator
{
private:
    double left, right, lower, upper;
    int width, height, iters;
    std::unique_ptr<int[]> image, local_buffer;
    int rank, size;
    int rows_per_process, local_rows;

    void computeLocalRows()
    {
        double x_offset = (right - left) / width;
        double y_offset = (upper - lower) / height;

        // Calculate which rows this process handles
        int start_row = rank * rows_per_process;
        int end_row = (rank == size - 1) ? height : (rank + 1) * rows_per_process;
        local_rows = end_row - start_row;

        // Allocate local buffer for this process's rows
        local_buffer = std::make_unique<int[]>(local_rows * width);

        // Constants for vectorized computation
        __m512d vec_two = _mm512_set1_pd(2.0);
        __m512d vec_four = _mm512_set1_pd(4.0);
        __m512d vec_x_offset = _mm512_set1_pd(x_offset);
        __m512d vec_left = _mm512_set1_pd(left);

        // Process assigned rows
        #pragma omp parallel for schedule(dynamic, 1)
        for (int local_j = 0; local_j < local_rows; local_j++) 
        {
            int global_j = start_row + local_j;
            double y0 = global_j * y_offset + lower;
            __m512d vec_y0 = _mm512_set1_pd(y0);

            int i;
            for (i = 0; i < width - 7; i += 8)
            {
                // Vectorized computation
                __m512d vec_x0 = _mm512_fmadd_pd(
                    _mm512_set_pd(i+7, i+6, i+5, i+4, i+3, i+2, i+1, i),
                    vec_x_offset,
                    vec_left
                );

                __m512d vec_x = _mm512_setzero_pd();
                __m512d vec_y = _mm512_setzero_pd();
                __m512d vec_x2 = _mm512_setzero_pd();
                __m512d vec_y2 = _mm512_setzero_pd();
                __m512d vec_length_squared = _mm512_setzero_pd();
                __m256i vec_repeats = _mm256_setzero_si256();
                __mmask8 mask = 0xFF;

                for (int iter = 0; iter < iters; ++iter) {
                    vec_x2 = _mm512_mul_pd(vec_x, vec_x);
                    vec_y2 = _mm512_mul_pd(vec_y, vec_y);
                    vec_length_squared = _mm512_add_pd(vec_x2, vec_y2);
                    mask = _mm512_cmp_pd_mask(vec_length_squared, vec_four, _CMP_LT_OS);
                    if (!mask) break;

                    __m512d vec_2xy = _mm512_mul_pd(_mm512_mul_pd(vec_x, vec_y), vec_two);
                    vec_x = _mm512_add_pd(_mm512_sub_pd(vec_x2, vec_y2), vec_x0);
                    vec_y = _mm512_add_pd(vec_2xy, vec_y0);

                    vec_repeats = _mm256_mask_add_epi32(
                        vec_repeats,
                        mask,
                        vec_repeats,
                        _mm256_set1_epi32(1)
                    );
                }
                
                // Store in local buffer instead of global image
                _mm256_storeu_epi32(&local_buffer[local_j * width + i], vec_repeats);
            }

            // Handle remaining pixels
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
                local_buffer[local_j * width + i] = repeats;
            }
        }
    }

    void gatherResults() {
        if (rank == 0) {
            // Root process: first copy its own data
            std::memcpy(image.get(), local_buffer.get(), local_rows * width * sizeof(int));
            
            // Then receive data from other processes
            for (int src = 1; src < size; src++) {
                // Calculate number of rows for this source process
                int src_start_row = src * rows_per_process;
                int src_end_row = (src == size - 1) ? height : (src + 1) * rows_per_process;
                int src_rows = src_end_row - src_start_row;
                
                // Receive data from source process
                MPI_Recv(
                    &image[src_start_row * width],  // Receive buffer (offset to correct position)
                    src_rows * width,               // Number of elements to receive
                    MPI_INT,                        // Data type
                    src,                            // Source rank
                    0,                              // Message tag
                    MPI_COMM_WORLD,                 // Communicator
                    MPI_STATUS_IGNORE               // Status (ignored)
                );
            }
        } else {
            // Other processes: send their local results to root
            MPI_Send(
                local_buffer.get(),     // Send buffer
                local_rows * width,     // Number of elements to send
                MPI_INT,                // Data type
                0,                      // Destination rank (root)
                0,                      // Message tag
                MPI_COMM_WORLD          // Communicator
            );
        }
    }

public:
    MandelbrotGenerator(double l, double r, double low, double up, int w, int h, int iters)
        : left(l), right(r), lower(low), upper(up), width(w), height(h), iters(iters)
    {
        // Get MPI rank and size
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Local buffer for this process's calculations
        if (rank == 0) image = std::make_unique<int[]>(width * height);

        // Calculate rows per process
        rows_per_process = height / size;
        if (rows_per_process == 0) {
            throw std::runtime_error("More processes than rows");
        }
    }

    void generate()
    {
        nvtx3::scoped_range range("Calculation");
        // Compute local portion
        computeLocalRows();
        
        // Gather results to root process
        gatherResults();
        
        // Ensure all processes are synchronized
        MPI_Barrier(MPI_COMM_WORLD);
    }

    void saveToPNG(const std::string& filename) const
    {
        if (rank == 0) {
            PNGWriter writer(filename, iters, width, height, image.get());
            writer.write();
        }
    }
};

int main(int argc, char** argv) {
    try
    {   // Detect available CPUs
        // cpu_set_t cpu_set;
        // sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
        // std::cout << CPU_COUNT(&cpu_set) << " cpus available\n";

        // Initialize MPI
        MPI_Init(&argc, &argv);

        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);

        // Validate arguments
        if (argc != 9) {
            if (rank == 0) {
                throw std::runtime_error("Invalid number of arguments");
            }
            MPI_Finalize();
            return 1;
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

        // Generate and save Mandelbrot set
        MandelbrotGenerator mandelbrot(left, right, lower, upper, width, height, iters);
        mandelbrot.generate();
        mandelbrot.saveToPNG(filename);
        
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