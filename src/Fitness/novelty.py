import neat
from typing import List

def novelty_fitness(genomes: List[neat.DefaultGenome], k:int):
    weight_coef = 0.5
    disjoint_coef = 1
    for i in range(len(genomes) - 1):
        for b in genomes[i+1:]:
            print(distance(genomes[i][1], b[1], weight_coef, disjoint_coef))
            

def distance(a, b, weight_coef, disjoint_coef):
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
                    node_distance += gene_distance(n1, n2, weight_coef)

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
                    connection_distance += conn_distance(c1, c2, weight_coef)

            max_conn = max(len(a.connections), len(b.connections))
            connection_distance = (connection_distance +
                                   (disjoint_coef *
                                    disjoint_connections)) / max_conn

        distance = node_distance + connection_distance
        return distance

def conn_distance(a, b, weight_coef):
    d = abs(a.weight - b.weight)
    if a.enabled != b.enabled:
        d += 1.0
    return d * weight_coef

def gene_distance(a, b, weight_coef):
    d = abs(a.bias - b.bias) + abs(a.response - b.response)
    if a.activation != b.activation:
        d += 1.0
    if a.aggregation != b.aggregation:
        d += 1.0
    return d * weight_coef