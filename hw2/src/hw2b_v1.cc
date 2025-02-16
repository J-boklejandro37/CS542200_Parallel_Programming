// static load, no vectorization -> Time: 346.93 s

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <assert.h>
#include <cstring>
#include <iostream>
#include <memory>
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
    int width, height, iterations;
    std::unique_ptr<int[]> image, local_image;
    int rank, size, rows_per_process, start_row, end_row, local_size;

public:
    MandelbrotGenerator(double l, double r, double low, double up, int w, int h, int iters, int rank, int size)
        : left(l), right(r), lower(low), upper(up), width(w), height(h), iterations(iters), rank(rank), size(size)
    {
        // Calculate loca range for this MPI process
        rows_per_process = height / size;
        start_row = rank * rows_per_process;
        end_row = (rank == size - 1) ? height : start_row + rows_per_process;
        local_size = (end_row - start_row) * width;

        // Local buffer for this process's calculations
        local_image = std::make_unique<int[]>(local_size);
    }

    void generate()
    {   // OpenMP parallel region for rows
        #pragma omp parallel for schedule(dynamic) collapse(2)
        for (int j = start_row; j < end_row; ++j)
        {   
            for (int i = 0; i < width; ++i)
            {
                double y0 = j * ((upper - lower) / height) + lower;
                double x0 = i * ((right - left) / width) + left;

                int repeats = 0;
                double x = 0;
                double y = 0;
                double length_squared = 0;
                while (repeats < iterations && length_squared < 4)
                {
                    double temp = x * x - y * y + x0;
                    y = 2 * x * y + y0;
                    x = temp;
                    length_squared = x * x + y * y;
                    ++repeats;
                }
                local_image[(j - start_row) * width + i] = repeats;
            }
        }

        // Gather results from all processes
        if (rank == 0)
        {   // Collect memory for image
            image = std::make_unique<int[]>(width * height);
            // Copy local results to the main image array
            std::copy(local_image.get(), local_image.get() + local_size, 
                     image.get() + start_row * width);

            // Receive results from other processes
            for (int r = 1; r < size; r++)
            {
                int r_start = r * rows_per_process;
                int r_rows = (r == size - 1) ? (height - r_start) : rows_per_process;
                MPI_Recv(image.get() + r_start * width, 
                        r_rows * width, MPI_INT, r, 0, 
                        MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            }
        }
        else
        {   // Send local results to rank 0
            MPI_Send(local_image.get(), local_size, 
                    MPI_INT, 0, 0, MPI_COMM_WORLD);
        }
    }

    void saveToPNG(const std::string& filename) const
    {
        PNGWriter writer(filename, iterations, width, height, image.get());
        writer.write();
    }
};

int main(int argc, char** argv) {
    try
    {   // Detect available CPUs
        cpu_set_t cpu_set;
        sched_getaffinity(0, sizeof(cpu_set), &cpu_set);
        std::cout << CPU_COUNT(&cpu_set) << " cpus available\n";

        // Validate arguments
        if (argc != 9) throw std::runtime_error("Invalid number of arguments");

        // Initialize MPI
        MPI_Init(&argc, &argv);
        int rank, size;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        MPI_Comm_size(MPI_COMM_WORLD, &size);

        // Parse arguments
        const std::string filename = argv[1];
        int iterations = std::stoi(argv[2]);
        double left = std::stod(argv[3]);
        double right = std::stod(argv[4]);
        double lower = std::stod(argv[5]);
        double upper = std::stod(argv[6]);
        int width = std::stoi(argv[7]);
        int height = std::stoi(argv[8]);

        // Handing arbitrary number of processes
        MPI_Group orig_group, new_group;
        MPI_Comm NEW_COMM = MPI_COMM_WORLD;
        if (height < size)
        {   // Extract the original group handle
            MPI_Comm_group(MPI_COMM_WORLD, &orig_group);

            // Define range as [start, end, stride], it can accept multiple range groups, hence a two dimensional array
            int range[1][3] = {{0, height - 1, 1}};

            // Create new group with processes 0 to (height-1)
            MPI_Group_range_incl(orig_group, 1, range, new_group);

            // Create new communicator, those not get included will be assigned NEW_COMM = MPI_COMM_NULL
            MPI_Comm_create(MPI_COMM_WORLD, new_group, &NEW_COMM);

            // Terminate those not getting included
            if (NEW_COMM == MPI_COMM_NULL)
            {
                MPI_Finalize();
                return 0;
            }
        }

        // Generate and save Mandelbrot set
        MandelbrotGenerator mandelbrot(left, right, lower, upper, width, height, iterations, rank, size);
        mandelbrot.generate();
        if (rank == 0)
            mandelbrot.saveToPNG(filename);
        
        MPI_Finalize();

        return 0;
    } 
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}