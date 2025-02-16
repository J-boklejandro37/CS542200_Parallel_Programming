// Dijkstra's algo with FibonacciHeap(decrease key) instead of priority_queue -> 38.96 s
// In theory, Fibonacci heaps offer better symptotic complexity(O(E + V log V) vs O((V + E) log V) with binary heaps)
// However, in practice, binary heaps has better constant factors(simpler implementation), better cache locality(contiguous in memory)
// The theoretical advantage of Fibonacci heaps only becomes noticeable with extremely large, dense graphs.

#include <algorithm>
#include <cmath>
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

template <typename T>
class FibonacciHeap {
private:
    struct Node {
        T key;
        int vertex;
        int degree;
        bool marked;
        Node* parent;
        Node* child;
        Node* left;
        Node* right;
        
        Node(T k, int v) : key(k), vertex(v), degree(0), marked(false),
                          parent(nullptr), child(nullptr) {
            left = right = this;
        }
    };
    
    Node* min;
    std::vector<Node*> nodes;
    int n;
    
    void link(Node* y, Node* x) {
        // Remove y from root list
        y->left->right = y->right;
        y->right->left = y->left;
        
        // Make y a child of x
        y->parent = x;
        if (!x->child) {
            x->child = y;
            y->right = y;
            y->left = y;
        } else {
            y->left = x->child;
            y->right = x->child->right;
            x->child->right = y;
            y->right->left = y;
        }
        
        x->degree++;
        y->marked = false;
    }
    
    void consolidate() {
        std::vector<Node*> A(log2(n) * 2, nullptr);
        
        // Create root list
        std::vector<Node*> roots;
        if (!min) return;
        
        Node* current = min;
        do {
            roots.push_back(current);
            current = current->right;
        } while (current != min);
        
        for (Node* w : roots) {
            Node* x = w;
            int d = x->degree;
            
            while (A[d]) {
                Node* y = A[d];
                if (x->key > y->key) std::swap(x, y);
                
                link(y, x);
                A[d] = nullptr;
                d++;
            }
            A[d] = x;
        }
        
        min = nullptr;
        for (Node* a : A) {
            if (a) {
                if (!min) {
                    min = a;
                    a->left = a->right = a;
                } else {
                    a->left = min;
                    a->right = min->right;
                    min->right = a;
                    a->right->left = a;
                    if (a->key < min->key) min = a;
                }
            }
        }
    }
    
    void cut(Node* x, Node* y) {
        // Remove x from child list of y
        if (x->right == x) {
            y->child = nullptr;
        } else {
            y->child = x->right;
            x->right->left = x->left;
            x->left->right = x->right;
        }
        y->degree--;
        
        // Add x to root list
        x->right = min->right;
        x->left = min;
        min->right = x;
        x->right->left = x;
        x->parent = nullptr;
        x->marked = false;
    }
    
    void cascadingCut(Node* y) {
        Node* z = y->parent;
        if (z) {
            if (!y->marked) {
                y->marked = true;
            } else {
                cut(y, z);
                cascadingCut(z);
            }
        }
    }

public:
    FibonacciHeap(int size) : min(nullptr), n(0) {
        nodes.resize(size, nullptr);
    }
    
    ~FibonacciHeap() {
        for (auto node : nodes) delete node;
    }
    
    void push(int vertex, T key) {
        Node* x = nodes[vertex];
        if (!x) {
            x = new Node(key, vertex);
            nodes[vertex] = x;
            n++;
            
            if (!min) {
                min = x;
            } else {
                x->right = min->right;
                x->left = min;
                min->right = x;
                x->right->left = x;
                if (key < min->key) min = x;
            }
        } else {
            decreaseKey(vertex, key);
        }
    }
    
    std::pair<T, int> pop() {
        Node* z = min;
        if (!z) throw std::runtime_error("Heap is empty");
        
        if (z->child) {
            Node* child = z->child;
            Node* current = child;
            do {
                Node* next = current->right;
                current->right = min->right;
                current->left = min;
                min->right = current;
                current->right->left = current;
                current->parent = nullptr;
                current = next;
            } while (current != child);
        }
        
        z->left->right = z->right;
        z->right->left = z->left;
        
        if (z == z->right) {
            min = nullptr;
        } else {
            min = z->right;
            consolidate();
        }
        
        std::pair<T, int> result = {z->key, z->vertex};
        nodes[z->vertex] = nullptr;
        delete z;
        n--;
        return result;
    }
    
    void decreaseKey(int vertex, T key) {
        Node* x = nodes[vertex];
        if (!x) return;
        if (key > x->key) return;
        
        x->key = key;
        Node* y = x->parent;
        
        if (y && x->key < y->key) {
            cut(x, y);
            cascadingCut(y);
        }
        
        if (x->key < min->key) min = x;
    }
    
    bool empty() const {
        return min == nullptr;
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

    // std::vector<int> dijkstra(int source)
    // {
    //     std::vector<int> dist(n, INF);
    //     FibonacciHeap heap(n);
        
    //     dist[source] = 0;
    //     heap.push(source, 0);
        
    //     while (!heap.empty()) {
    //         auto [d, u] = heap.pop();
            
    //         for (const Edge& edge : graph[u]) {
    //             int v = edge.dest;
    //             int weight = edge.weight;
                
    //             if (dist[u] + weight < dist[v]) {
    //                 dist[v] = dist[u] + weight;
    //                 heap.decrease_key(v, dist[v]);
    //             }
    //         }
    //     }
        
    //     return dist;
    // }
    std::vector<int> dijkstra(int source) {
        std::vector<int> dist(n, INF);
        FibonacciHeap<int> heap(n);
        
        dist[source] = 0;
        heap.push(source, 0);
        
        while (!heap.empty()) {
            auto [d, u] = heap.pop();
            
            for (const Edge& edge : graph[u]) {
                int v = edge.dest;
                int weight = edge.weight;
                
                if (dist[u] + weight < dist[v]) {
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