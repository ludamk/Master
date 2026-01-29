# fire
#
import numpy as np
from scipy.sparse.csgraph import dijkstra
from scipy import sparse

class Fire:
    def __init__(self, g, initialFire, points, burned=None):
        # Graph
        self.g = g
        
        # Numpy of points
        self.points = points
        
        # list of nodes on initialFire (list of indexes) and burned (boolean)
        self.initialFire = initialFire
        self.fireFront   = initialFire
        
        # Contain history of firefront (contour nodes)
        self.fireFrontHist = []
        
        if burned is None:
            burned = np.zeros(g.size, dtype=bool)
        
        # Set unvisited and distances (boolean and float)
        self.unvisited = ~burned
        
        distances              = np.full(g.size, np.inf)
        distances[initialFire] = 0
        self.distances         = distances
        
        # Contour Lines and contour times
        self.contourLinesSegments2D = []
        self.contourT2D = []
        
        self.contourLinesSegments3D = []
        self.contourT3D = []
        
        
    def findFireFront(self, t, extinguishFactor=np.inf):
        distances = self.distances
        
        # All burned nodes and fireFront candidates
        burned_mask = distances <= t
        fireFront_candidate_mask = burned_mask & (distances >= t - extinguishFactor)
    
        # Get CSR arrays
        adj_csr = self.g.adj_matrix.tocsr()
        indptr = adj_csr.indptr
        indices = adj_csr.indices
    
        # Find candidate for boolean index for neighbors to fireFront
        not_burned_mask = distances > t
    
        # Results
        fireFront_nodes = []
    
        # Loop over candidates and find indices
        candidate_indices = np.nonzero(fireFront_candidate_mask)[0]
    
        for b in candidate_indices:
            start = indptr[b]
            end = indptr[b + 1]
            neighbors = indices[start:end]

            # Check if any neighbors are not burned = b is fireFront node
            if np.any(not_burned_mask[neighbors]):
                fireFront_nodes.append(b)
    
        if fireFront_nodes:
            fireFront_nodes = np.array(fireFront_nodes, dtype=int)
            self.fireFront = fireFront_nodes
        else:
            self.fireFront = np.array([], dtype=int)
        
        return self.fireFront

        
    def generateFire(self, T, extinguishFactor=np.inf):
        # T: Time to stop the fire
        # extinguishFactor: Scalar that tells when the firefront is extinguished
    
        if np.ndim(T) == 0:
            t = T
        else:
            t = T[-1]
        
        # Get adj_matrix to csr format, better suitable for dijkstra
        adj_matrix_csr = self.g.adj_matrix.tocsr()
        
        # adj_matrix without edge >= extinguishFactor
        coo = adj_matrix_csr.tocoo()
        keep = coo.data < extinguishFactor
        coo2 = sparse.coo_matrix((coo.data[keep], (coo.row[keep], coo.col[keep])), shape=adj_matrix_csr.shape)
        adj_matrix_dijkstra = coo2.tocsr()

        # limit = t + 5 (5 is an error term I have added)
        distances, predecessors, _ = dijkstra(adj_matrix_dijkstra, directed=True, indices=self.initialFire, return_predecessors=True, limit=(t+5), min_only=True)
        
        # Update fire
        self.distances = distances
        self.unvisited = distances == np.inf
        self.burned    = ~self.unvisited

        self.fireFront = self.findFireFront(t, extinguishFactor)
        