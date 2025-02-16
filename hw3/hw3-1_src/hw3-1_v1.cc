// vector<vector<int>> Dist -> [19/21] 95.04 s
// + omp -> [21/21] 89.87 s

#include <algorithm>
#include <fstream>
#include <iostream>
#include <limits>
#include <omp.h>
#include <stdexcept>
#include <string>
#include <vector>

#define ceil(a, b) ((a + b - 1) / b)

const int INF = (1 << 30) - 1;
const int V = 50010;

class FloydWarshall {
private:
    int n, m;
    std::vector<std::vector<int>> Dist;  // Changed to vector

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
        Dist.resize(n, std::vector<int>(n, INF));
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
        {
            Dist[i][i] = 0;  // Initialize diagonal
        }

        // Read edges
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
        if (!file.is_open())
            throw std::runtime_error("Could not open output file.");

        // Write data
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                Dist[i][j] = std::min(Dist[i][j], INF);
            }
            file.write(reinterpret_cast<const char*>(Dist[i].data()), sizeof(int) * n);
        }

        file.close();
    }

    void compute()
    {
        for (int k = 0; k < n; k++)
        {
            #pragma omp parallel for schedule(static) collapse(2)
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

int main(int argc, char* argv[]) {
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