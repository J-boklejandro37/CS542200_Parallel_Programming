// Time: 714.24 s

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#define PNG_NO_SETJMP
#include <sched.h>
#include <assert.h>
#include <png.h>
#include <iostream>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <pthread.h>

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
                        color[1] = color[2] = p % 16 * 16;
                    }
                    else color[0] = p % 16 * 16;
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
    int width, height, iterations, ncpus;
    std::unique_ptr<int[]> image;

    struct ThreadData
    {
        MandelbrotGenerator* obj;
        int thread_id;
    };

    static void* wrapper(void* arg) // need to use static so that the type can be void* (*)(void*), or it will be void* (MandelbrotGenerator::*)(void*), meaning a "member function pointer"
    {
        ThreadData* data = static_cast<ThreadData*>(arg);
        data->obj->compute(data->thread_id);
        return nullptr;
    }

    void compute(int thread_id)
    {
        int rows_per_thread = height / ncpus;
        int extra = height % ncpus;
        int start_row = thread_id * rows_per_thread + std::min(thread_id, extra);
        int end_row = start_row + rows_per_thread + (thread_id < extra ? 1 : 0);

        for (int j = start_row; j < end_row; ++j)
        {
            double y0 = j * ((upper - lower) / height) + lower;
            for (int i = 0; i < width; ++i)
            {
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
                image[j * width + i] = repeats;
            }
        }
    }

public:
    MandelbrotGenerator(double l, double r, double low, double up, int w, int h, int iters, int ncpus)
        : left(l), right(r), lower(low), upper(up), width(w), height(h), iterations(iters), ncpus(ncpus)
    {
        image = std::make_unique<int[]>(width * height);
    }

    void generate()
    {
        pthread_t threads[ncpus]; // for thread instances
        int* tids = new int[ncpus]; // for thread_id
        ThreadData* thread_data = new ThreadData[ncpus]; // for callback function's argument

        // create threads
        for (int i = 0; i < ncpus; ++i)
        {
            tids[i] = i;
            thread_data[i] = {this, i}; // pass in "this" so that the static function "wrapper" can access the member function "compute"
            pthread_create(&threads[i], nullptr, wrapper, &thread_data[i]); // can only pass in static function as callback function
        }

        // wait for all
        for (int i = 0; i < ncpus; ++i)
        {
            pthread_join(threads[i], nullptr);
        }

        delete[] tids;
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
        int ncpus = CPU_COUNT(&cpu_set);
        std::cout << ncpus << " cpus available\n";

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
        MandelbrotGenerator mandelbrot(left, right, lower, upper, width, height, iterations, ncpus);
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