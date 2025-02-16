// Dijkstra's algo with priority_queue
// [testcase c18.1]
// -> 36.09 s (nothing)
// ->  3.53 s (omp, schedule(static))
// ->  3.58 s (omp, schedule(static,1))
// ->  3.48 s (omp, schedule(dynamic,1))
// ->  3.53 s (omp, schedule(guided,1))

// [Full testcase]
// -> 41.77 s (omp, schedule(guided,1))
// -> 40.96 s (omp, schedule(dynamic,1))

#include <algorithm>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <queue>
#include <string>
#include <stdexcept>
#include <vector>

const int INF = (1 << 30) - 1;

struct Edge
{
    int dest;
    int weight;
    Edge(int d, int w) : dest(d), weight(w) {}
};

struct Node
{
    int vertex;
    int distance;
    Node(int v, int d) : vertex(v), distance(d) {}
    bool operator>(const Node& other) const
    {
        return distance > other.distance;
    }
};

class DijkstraAllPairs
{
private:
    int n, m;
    std::vector<std::vector<int>> Dist;
    std::vector<std::vector<Edge>> graph;

public:
    DijkstraAllPairs() : n(0), m(0) {}

    void input(const std::string& filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("Could not open input file.");

        file.read(reinterpret_cast<char*>(&n), sizeof(int));
        file.read(reinterpret_cast<char*>(&m), sizeof(int));

        // Initialize distance matrix and graph
        Dist.resize(n, std::vector<int>(n, INF));
        graph.resize(n);
        
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++)
        {
            Dist[i][i] = 0;
        }

        // Read edges
        for (int idx = 0; idx < m; idx++)
        {
            int edge[3]; // from, to, weight
            file.read(reinterpret_cast<char*>(edge), sizeof(int) * 3);
            graph[edge[0]].emplace_back(edge[1], edge[2]);
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

    std::vector<int> dijkstra(int source)
    {
        std::vector<int> dist(n, INF);
        std::priority_queue<Node, std::vector<Node>, std::greater<Node>> pq;
        
        dist[source] = 0;
        pq.emplace(source, 0);
        
        while (!pq.empty())
        {
            int u = pq.top().vertex;
            int d = pq.top().distance;
            pq.pop();
            
            if (d > dist[u]) continue;  // The shortest path has been updated
            
            for (const Edge& edge : graph[u])
            {
                int v = edge.dest;
                int weight = edge.weight;
                
                if (dist[u] + weight < dist[v])
                {
                    dist[v] = dist[u] + weight;
                    pq.emplace(v, dist[v]);
                }
            }
        }
        
        return dist;
    }

    void compute()
    {
        #pragma omp parallel for schedule(dynamic, 1)
        for (int i = 0; i < n; i++)
        {
            std::vector<int> distances = dijkstra(i);
            for (int j = 0; j < n; j++)
            {
                Dist[i][j] = distances[j];
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
        DijkstraAllPairs ap;
        ap.input(argv[1]);
        ap.compute();
        ap.output(argv[2]);
    }
    catch (const std::exception& e)
    {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}