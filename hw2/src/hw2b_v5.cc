// dynamic load balancing (chunk size 60), vectorization optimized, omp schedule(dynamic, 1) -> 93.77 s, (20241031) 99.59 s, (20241101) 95.98 s

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

// Define MPI message tags
enum Tags
{
    WORK_REQUEST = 1,
    WORK_DISPATCH = 2,
    RESULT = 3,
    TERMINATE = 4
};

// For passing the work
struct WorkUnit
{
    int start_row;
    int num_rows;
    WorkUnit() {}
    WorkUnit(int s, int n) : start_row(s), num_rows(n) {}
};

// For passing the result
struct ResultUnit
{
    int start_row;
    int num_rows;
    std::vector<int> data;
    ResultUnit() {}
    ResultUnit(int s, int n) : start_row(s), num_rows(n) {}
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
    std::queue<WorkUnit> work_queue;
    for (int row = 0; row < height; row += CHUNK_SIZE)
    {
        int num_rows = std::min(CHUNK_SIZE, height - row);
        work_queue.emplace(row, num_rows);
    }

    // Track finished workers
    int finished_workers = 0;

    // Create datatype for WorkUnit
    MPI_Datatype WORK_UNIT_TYPE;
    MPI_Type_contiguous(2, MPI_INT, &WORK_UNIT_TYPE); // Likey a reserved space for buffer
    MPI_Type_commit(&WORK_UNIT_TYPE); // Likely registering the address of the buffer

    // Waiting for work requests or results until all works are done
    // Need to make sure all processes are terminated
    MPI_Status status;
    while (!work_queue.empty() || finished_workers < num_processes - 1)
    {
        // Probe only check if there's incoming message, do not receive the data
        // If true, the flag is set to 1, status is stored in status
        // This procedure automatically deal with race condition
        // There are two implementation, one is the currently using one.
        // The other is to use MPI_Iprobe(..., &flag, &statue); if (flag) {}, will keep entering the while loop though.
        MPI_Probe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);

        int source = status.MPI_SOURCE;

        switch (status.MPI_TAG)
        {
            case WORK_REQUEST:
            {
                // Receiving the request (so that sender can move on, no actual data receiving here)
                int dummy;
                MPI_Recv(&dummy, 1, MPI_INT, source, WORK_REQUEST, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Dispatch the work if exists, or terminate the process
                if (!work_queue.empty())
                {
                    auto unit = work_queue.front();
                    work_queue.pop();
                    MPI_Send(&unit, 1, WORK_UNIT_TYPE, source, WORK_DISPATCH, MPI_COMM_WORLD);
                }
                else
                {   // Sent the terminate signal
                    MPI_Send(nullptr, 0, MPI_INT, source, TERMINATE, MPI_COMM_WORLD);
                    finished_workers += 1;
                }
                break;
            }
            case RESULT:
            {
                // Get the metadata (start_row & num_rows) first
                ResultUnit result;
                MPI_Recv(&result.start_row, 2, MPI_INT, source, RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                // Then receive the actual data
                result.data.resize(result.num_rows * width);
                MPI_Recv(result.data.data(), result.data.size(), MPI_INT, source, RESULT, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

                // Store onto image
                master.storeRows(result.start_row, result.data, result.num_rows);

                break;
            }
        }
    }

    // Save the final image
    master.saveToPNG(filename);

    // Clean up MPI datatype
    MPI_Type_free(&WORK_UNIT_TYPE);
}

void workerProcess(int rank, double left, double right, double lower, double upper, int width, int height, int iters)
{
    Worker worker(left, right, lower, upper, width, height, iters);

    // Create datatype for WorkUnit
    MPI_Datatype WORK_UNIT_TYPE;
    MPI_Type_contiguous(2, MPI_INT, &WORK_UNIT_TYPE); // Likey a reserved space for buffer
    MPI_Type_commit(&WORK_UNIT_TYPE); // Likely registering the address of the buffer

    // Keep requesting for work until getting a termination, then break
    while (true)
    {
        // Request work from master
        int dummy = 0;
        MPI_Send(&dummy, 1, MPI_INT, 0, WORK_REQUEST, MPI_COMM_WORLD);
        
        // Receive response from master, could be Terminate or WorkDispatch
        MPI_Status status;
        MPI_Probe(0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
        
        if (status.MPI_TAG == TERMINATE)
        {
            // Break if getting terminated
            MPI_Recv(&dummy, 0, MPI_INT, 0, TERMINATE, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            break;
        }
        
        // Receive work unit
        WorkUnit unit;
        MPI_Recv(&unit, 1, WORK_UNIT_TYPE, 0, WORK_DISPATCH, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        
        // Process the work unit
        std::vector<int> result = worker.computeRows(unit.start_row, unit.num_rows);
        
        // Prepare and send result
        // First send header (start_row & num_rows)
        int header[2] = {unit.start_row, unit.num_rows};
        MPI_Send(header, 2, MPI_INT, 0, RESULT, MPI_COMM_WORLD);
        
        // Then send the computed data
        MPI_Send(result.data(), result.size(), MPI_INT, 0, RESULT, MPI_COMM_WORLD);
    }
    
    // Clean up MPI datatype
    MPI_Type_free(&WORK_UNIT_TYPE);
}