import neat
from typing import List
import numpy as np 
import math
from src.logger import NoveltyLogger
import collections

class Novelty:
    def __init__(self, novelty_log_path, threshold=0.85, overwrite=True, squash_function=True) -> None:
        self.archive = None
        self.threshold = threshold
        if overwrite:
            self.logger = NoveltyLogger(novelty_log_path, "novelty.csv", header=["KNN Novelty", "Archive Novelty"])
        else:
            self.logger = NoveltyLogger(novelty_log_path, "novelty.csv")
        
        if squash_function:
            self.squash_func = self.__squash_function
        else:
            self.squash_func = lambda x: x
            
    def novelty_score(self, genomes: List[neat.DefaultGenome], k:int):
        # Novelty config
        weight_coef = 0.5 
        disjoint_coef = 1 
        
        # Get novelty within population to n_pop/3 nearest neightbours
        novelty = self.knn(genomes, math.ceil(2*len(genomes)/3), weight_coef, disjoint_coef)
        log = [[i,-1] for i in novelty]
                
        # Check if archive is empty or not
        if self.archive is not None:
            # Add archived novelty to population novelty
            arch = self.archived_dist(genomes, weight_coef, disjoint_coef)
            for i in range(len(novelty)):
                novelty[i] = (arch[i] + novelty[i])/2
                log[i][1] = arch[i]

        # Log novelty values
        self.logger.log_iteration(log)
        
        # Archive individuals above novelty threshold 
        g_arch = []
        for (__, g), nov in zip(genomes, novelty):
            if nov > self.threshold:
                g_arch.append(g)
        if len(g_arch) > 0:
            self.archive = g_arch
        else:
            self.archive = None
            
        return novelty        
        
    def archived_dist(self, genomes:List[neat.DefaultGenome], weight_coef:float, disjoint_coef:float)\
        -> List[float]:
        genomes = [g for __, g in genomes]
        # Get KNN to all archived genomes
        k = math.ceil(len(self.archive) / 3)
        dist = []
        for genome in genomes:
            dist.append(self.single_knn(genome, self.archive, k, weight_coef, disjoint_coef))
        return dist
    
    def knn(self, genomes:List[neat.DefaultGenome], k:int, weight_coef:float, disjoint_coef:float) \
        -> List[float]:
        assert type(k) == int
        assert k < len(genomes)
        out = []
        # Reindex genomes
        g = [(i, g) for i, (__, g) in enumerate(genomes)]
        # Store distances between each genome
        dist = np.zeros((len(genomes), len(genomes)))
        # Get distances between each genome
        for ai, a in enumerate(g[:-1]):
            for b in g[ai + 1:]:
                ab_dist = self.squash_func(self.distance(a[1], b[1], weight_coef, disjoint_coef))
                dist[a[0]][b[0]] = ab_dist
                dist[b[0]][a[0]] = ab_dist

        # Get knn for each genome
        for i, a in enumerate(dist):
            neighs = sorted(list(a[:i]) + list(a[i+1:]))
            out.append(np.average(neighs[:k]))
        return out            
            
    def single_knn(self, a:neat.DefaultGenome, b:List[neat.DefaultGenome], k:int, weight_coef:float,\
                   disjoint_coef:float) -> float:
        dist = []
        for bi in b:
            dist.append(self.squash_func(self.distance(a, bi, weight_coef, disjoint_coef)))
        dist = sorted(dist)[:k]
        return np.average(dist)
        

    def distance(self, a, b, weight_coef, disjoint_coef):
        """
        Returns the genetic distance between this genome and the b. This distance value
        is used to compute genome compatibility for speciation.
        """

        # Compute node gene distance component.
        node_distance = 0.0
        if a.nodes or b.nodes:
            disjoint_nodes = 0
            for k2 in b.nodes:
                if k2 not in a.nodes:
                    disjoint_nodes += 1

            for k1, n1 in a.nodes.items():
                n2 = b.nodes.get(k1)
                if n2 is None:
                    disjoint_nodes += 1
                else:
                    # Homologous genes compute their own distance value.
                    node_distance += self.gene_distance(n1, n2, weight_coef)

            max_nodes = max(len(a.nodes), len(b.nodes))
            node_distance = (node_distance +
                            (disjoint_coef *
                            disjoint_nodes)) / max_nodes

        # Compute connection gene differences.
        connection_distance = 0.0
        if a.connections or b.connections:
            disjoint_connections = 0
            for k2 in b.connections:
                if k2 not in a.connections:
                    disjoint_connections += 1

            for k1, c1 in a.connections.items():
                c2 = b.connections.get(k1)
                if c2 is None:
                    disjoint_connections += 1
                else:
                    # Homologous genes compute their own distance value.
                    connection_distance += self.conn_distance(c1, c2, weight_coef)

            max_conn = max(len(a.connections), len(b.connections))
            connection_distance = (connection_distance +
                                (disjoint_coef *
                                    disjoint_connections)) / max_conn

        distance = node_distance + connection_distance
        return distance

    def conn_distance(self, a, b, weight_coef):
        d = abs(a.weight - b.weight)
        if a.enabled != b.enabled:
            d += 1.0
        return d * weight_coef

    def gene_distance(self, a, b, weight_coef):
        d = abs(a.bias - b.bias) + abs(a.response - b.response)
        if a.activation != b.activation:
            d += 1.0
        if a.aggregation != b.aggregation:
            d += 1.0
        return d * weight_coef

    def __squash_function(self, x):
        return 1/(1+(math.e**((-6*x)+5)))
    
class DynamicNovelty():
    def __init__(self):
        self.buffer = collections.deque()
        self.window = 5
    
    def get_ratios(self, novelty_values, structure_values):
        assert type(novelty_values) == list and type(structure_values) == list
        assert len(novelty_values) == len(structure_values)
        
        p_nov, p_struct = self.__calculate_ratio(np.average(structure_values))

        out = []
        for i in range(len(novelty_values)):
            out.append(
                novelty_values[i]*p_nov + structure_values[i]*p_struct
            )
        return out    

    def __calculate_ratio(self, struct):
        if len(self.buffer) < self.window:
            # If there are not enough values in buffer, 
            # add and return 1:1 nov:struct
            self.buffer.appendleft(struct)
            return (0.5, 0.5)
        else:
            r_nov, r_struct = 0, 0 
            # If buffer is filled
            # Calculate average gradient
            avg_grad = np.average([abs(self.buffer[i] - self.buffer[i-1]) for i in range(1, len(self.buffer))])
            if avg_grad == 0:
                r_nov = 1
                r_struct = 0
            else:
                r_struct = avg_grad
                r_nov = 1 / avg_grad
                
            p_nov = r_nov / (r_nov + r_struct)
            p_struct = r_struct / (r_nov + r_struct)
            
            # Remove oldest value and add newest
            self.buffer.pop()
            self.buffer.appendleft(struct)
            
            return p_nov, p_struct
    
    