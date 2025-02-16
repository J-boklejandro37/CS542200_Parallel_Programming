// MPI static load (chunk by chunk), vectorization optimized, omp schedule(dynamic, 1) -> (20241101) 106.75 s

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

class MandelbrotGenerator
{
private:
    double left, right, lower, upper;
    int width, height, iters;
    int start_row, num_rows;
    std::shared_ptr<int[]> buffer;
    alignas(64) const double vec_init_sequence[8] = {0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0};

public:
    MandelbrotGenerator(double l, double r, double low, double up, int w, int h, int iters, int sr, int nr)
        : left(l), right(r), lower(low), upper(up), width(w), height(h), iters(iters), start_row(sr), num_rows(nr)
    {
        buffer = std::shared_ptr<int[]>(new int[width * num_rows]);
    }

    // For worker process
    std::shared_ptr<int[]> generate() const
    {
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
            for (i = 0; i < width - 7; i += 8)
            {
                // Vectorized computation
                __m512d vec_x0 = _mm512_fmadd_pd(
                    _mm512_add_pd(vec_sequence, _mm512_set1_pd(i)),
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
                    // Calculate y^2 first, then use fmadd for (x * x + y^2)
                    vec_x2 = _mm512_mul_pd(vec_x, vec_x);
                    vec_y2 = _mm512_mul_pd(vec_y, vec_y);
                    vec_length_squared = _mm512_fmadd_pd(vec_x, vec_x, vec_y2);

                    mask = _mm512_cmp_pd_mask(vec_length_squared, vec_four, _CMP_LT_OS);
                    if (!mask) break;

                    __m512d vec_2xy = _mm512_mul_pd(_mm512_mul_pd(vec_x, vec_y), vec_two);
                    vec_x = _mm512_add_pd(_mm512_sub_pd(vec_x2, vec_y2), vec_x0);
                    // vec_x = _mm512_fmadd_pd(vec_x, vec_x, _mm512_fnmadd_pd(vec_y, vec_y, vec_x0));
                    // vec_x = _mm512_fmadd_pd(vec_x, vec_x, _mm512_sub_pd(vec_x0, vec_y2));
                    vec_y = _mm512_add_pd(vec_2xy, vec_y0);

                    vec_repeats = _mm256_mask_add_epi32(
                        vec_repeats,
                        mask,
                        vec_repeats,
                        _mm256_set1_epi32(1)
                    );
                }
                
                // Store in local buffer instead of global image
                _mm256_storeu_epi32(&buffer[j * width + i], vec_repeats);
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
                buffer[j * width + i] = repeats;
            }
        }

        return buffer;
    }
};

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

        // Divide the work
        int remainder = height % size;
        int start_row = height / size * rank + std::min(rank, remainder);
        int num_rows = height / size + (rank < remainder);

        // Calculate Mandelbrot set
        MandelbrotGenerator mandelbrot(left, right, lower, upper, width, height, iters, start_row, num_rows);
        std::shared_ptr<int[]> buffer = mandelbrot.generate();

        // Prepare for gathering
        std::unique_ptr<int[]> image;
        if (rank == 0)
            image = std::make_unique<int[]>(width * height);

        // For MPI_Gatherv (arbitrary gather)
        int start_row_list[size];
        int num_rows_list[size];
        MPI_Gather(&start_row, 1, MPI_INT, start_row_list, 1, MPI_INT, 0, MPI_COMM_WORLD);
        MPI_Gather(&num_rows, 1, MPI_INT, num_rows_list, 1, MPI_INT, 0, MPI_COMM_WORLD);
        for (auto& x : start_row_list) x *= width;
        for (auto& x : num_rows_list) x *= width;

        MPI_Gatherv(buffer.get(), num_rows * width, MPI_INT,
                    image.get(), num_rows_list, start_row_list, MPI_INT,
                    0, MPI_COMM_WORLD);
        
        // Write to PNG
        if (rank == 0)
        {
            PNGWriter writer(filename, iters, width, height, image.get());
            writer.write();
        }
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