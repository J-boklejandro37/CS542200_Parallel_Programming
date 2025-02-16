// Uses class with Dist[V][V], which causes compile error.
// Even if static int Dist[V][V] works just fine, accessing it within class member will cause error.

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <limits>
#include <algorithm> // min, max
#include <stdexcept> // runtime_error

#define ceil(a, b) ((a + b - 1) / b)

const int INF = (1 << 30) - 1;
const int V = 50010;

static int Dist[V][V];

class FloydWarshall
{
private:
    int n, m;

public:
    FloydWarshall() : n(0), m(0) {}

    void input(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Could not open input file.");

        file.read(reinterpret_cast<char*>(&n), sizeof(int));
        file.read(reinterpret_cast<char*>(&m), sizeof(int));

        // Initialize distance matrix
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                if (i == j) Dist[i][j] = 0;
                else Dist[i][j] = INF;
            }
        }

        // // Read edges
        for (int idx = 0; idx < m; idx++)
        {
            int edge[3]; // from, to, weight
            file.read(reinterpret_cast<char*>(edge), sizeof(int) * 3);
            Dist[edge[0]][edge[1]] = edge[2];
        }

        file.close();
    }

    void output(const std::string& filename)
    {
        std::ofstream file(filename, std::ios::binary);
        if (!file)
            throw std::runtime_error("Could not open output file.");

        // Write data
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Dist[i][j] = std::min(Dist[i][j], INF);
            }
            file.write(reinterpret_cast<char*>(Dist[i]), sizeof(int) * n);
        }

        file.close();
    }

    void compute()
    {
        for (int k = 0; k < n; k++)
        {
            for (int i = 0; i < n; i++)
            {
                for (int j = 0; j < n; j++)
                {
                    Dist[i][j] = std::min(Dist[i][j], Dist[i][k] + Dist[k][j]);
                }
            }
        }
    }
};

int main(int argc, char* argv[])
{
    if (argc != 3)
    {
        std::cerr << "Usage: " << argv[0] << " <input_file> <output_file>" << std::endl;
        return 1;
    }

    try
    {
        FloydWarshall fw;
        fw.input(argv[1]);
        fw.compute();
        fw.output(argv[2]);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;

}