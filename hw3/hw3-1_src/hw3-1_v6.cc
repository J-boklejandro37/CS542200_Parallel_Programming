// Dijkstra's algo with BinaryHeap(decrease key) instead of priority_queue -> 33.99 s

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

class BinaryHeap {
private:
    std::vector<std::pair<int, int>> heap; // (distance, vertex)
    std::vector<int> pos; // Position of vertex in heap

    void siftUp(int idx) {
        while (idx > 0) {
            int parent = (idx - 1) / 2;
            if (heap[parent].first <= heap[idx].first) break;
            std::swap(heap[parent], heap[idx]);
            pos[heap[parent].second] = parent;
            pos[heap[idx].second] = idx;
            idx = parent;
        }
    }

    void siftDown(int idx) {
        while (true) {
            int smallest = idx;
            int left = 2 * idx + 1;
            int right = 2 * idx + 2;

            if (left < heap.size() && heap[left].first < heap[smallest].first)
                smallest = left;
            if (right < heap.size() && heap[right].first < heap[smallest].first)
                smallest = right;

            if (smallest == idx) break;

            std::swap(heap[idx], heap[smallest]);
            pos[heap[idx].second] = idx;
            pos[heap[smallest].second] = smallest;
            idx = smallest;
        }
    }

public:
    BinaryHeap(int n) : pos(n, -1) {}

    void push(int vertex, int distance) {
        if (pos[vertex] == -1) {
            heap.push_back({distance, vertex});
            pos[vertex] = heap.size() - 1;
            siftUp(heap.size() - 1);
        } else {
            int idx = pos[vertex];
            heap[idx].first = distance;
            siftUp(idx);
        }
    }

    std::pair<int, int> pop() {
        auto top = heap[0];
        pos[top.second] = -1;
        
        if (heap.size() > 1) {
            heap[0] = heap.back();
            pos[heap[0].second] = 0;
            heap.pop_back();
            siftDown(0);
        } else {
            heap.pop_back();
        }
        
        return top;
    }

    bool empty() const {
        return heap.empty();
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
        BinaryHeap heap(n);
        
        dist[source] = 0;
        heap.push(source, 0);
        
        while (!heap.empty())
        {
            auto [d, u] = heap.pop();
            
            for (const Edge& edge : graph[u])
            {
                int v = edge.dest;
                int weight = edge.weight;
                
                if (dist[u] + weight < dist[v])
                {
                    dist[v] = dist[u] + weight;
                    heap.push(v, dist[v]);
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