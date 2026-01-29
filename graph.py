# graph
#
from scipy.spatial import cKDTree
from scipy.sparse import lil_matrix
import heapq

def findNode(points, point):
    tree = cKDTree(points)
    dist, idx = tree.query(point)
    return idx

class weightedGraph:
    def __init__(self, size):
        self.adj_matrix = lil_matrix((size, size), dtype=float)
        self.size = size
        self.vertex_data = [''] * size

    def set_adj_matrix(self, mat):
        self.adj_matrix = mat
        self.size = mat.shape[0]
        self.vertex_data = [''] * self.size

    def update_adj_matrix(self, points, fun, metric="R"):
        # Updates weights in adj matrix
        rows, cols = self.adj_matrix.nonzero()
        
        # Update each weights
        for i, j in zip(rows, cols):
            if metric == "R":
                self.adj_matrix[i,j] = fun(points[i], points[j])

    def add_edge(self, u, v, weight, directed=True):
        self.adj_matrix[u, v] = weight
        if not directed:
            self.adj_matrix[v, u] = weight

    def add_vertex(self, data=''):
        self.size += 1
        self.adj_matrix.resize((self.size, self.size))
        self.vertex_data.append(data)

    def add_vertex_data(self, vertex, data):
        if 0 <= vertex < self.size:
            self.vertex_data[vertex] = data

    def dijkstra(self, start_vertex_data):
        start_vertex = self.vertex_data.index(start_vertex_data)
        distances = [float('inf')] * self.size
        predecessors = [None] * self.size
        distances[start_vertex] = 0
        visited = [False] * self.size

        heap = [(0, start_vertex)]

        while heap:
            current_distance, u = heapq.heappop(heap)
            if visited[u]:
                continue
            visited[u] = True

            # Adgang til naboer med sparse matrice
            neighbors = self.adj_matrix.rows[u]
            weights = self.adj_matrix.data[u]

            for idx, v in enumerate(neighbors):
                if visited[v]:
                    continue
                weight = weights[idx]
                alt = current_distance + weight
                if alt < distances[v]:
                    distances[v] = alt
                    predecessors[v] = u
                    heapq.heappush(heap, (alt, v))

        return distances, predecessors

    # Print dijkstra path
    def get_path(self, predecessors, start_vertex, end_vertex):
        path = []
        current = self.vertex_data.index(end_vertex)
        start_idx = self.vertex_data.index(start_vertex)
        while current is not None:
            path.insert(0, self.vertex_data[current])
            if current == start_idx:
                break
            current = predecessors[current]
        return '->'.join(path), path
    
    def copy(self):
        new_graph = weightedGraph(self.size)

        # kopiér adjacency matrix (deep copy)
        new_graph.adj_matrix = self.adj_matrix.copy()

        # kopiér vertex data
        new_graph.vertex_data = self.vertex_data.copy()

        return new_graph