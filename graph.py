import random
import numpy as np
from pkg_resources import resource_exists
from edge import *
from functions import *
from node import *
from operator import itemgetter
import networkx as nx
import matplotlib.pyplot as plt

class Graph:
    def __init__(self, nodes=[], edges=[]):
        self.nodes = [n for n in nodes]
        self.edges = [e for e in edges]
        self.weight = [0 for _ in edges]

    def add_node(self, node):
        self.nodes.append(node)

    def delete_node(self, node_idx):
        del self.nodes[node_idx]

    def add_edge(self, start, end):
        if start == end or self.edge_exists(start, end):
            return False
        self.edges.append(Edge(start, end))
        return True

    def add_edge_to_digraph(self, start, end):
        if start == end or self.edge_exists_in_digraph(start, end):
            return False
        self.edges.append(Edge(start, end))
        return True

    def add_nodes(self, amount):
        for i in range(amount):
            self.add_node(Node(i))

    def add_neighbour(self, node_index, neighbour):
        if node_index == neighbour or neighbour in self.nodes[
            node_index].neighbours:
            return False
        self.nodes[node_index].neighbours.append(neighbour)
        return True

    def add_neighbours(self, node1, node2):
        res = self.add_neighbour(node1, node2)
        res2 = self.add_neighbour(node2, node1)
        return res and res2

    def find_edge(self, idx1, idx2):
        for i, edge in enumerate(self.edges):
            if (edge.start == idx1 and edge.end == idx2) or (
                    edge.start == idx2 and edge.end == idx1):
                return edge, i
        return None, -1

    def find_edge_in_digraph(self, idx1, idx2):
        for i, edge in enumerate(self.edges):
            if (edge.start == idx1 and edge.end == idx2):
                return edge, i
        return None, -1

    def edge_exists(self, index1, index2):
        _, index = self.find_edge(index1, index2)
        return index >= 0

    def edge_exists_in_digraph(self, index1, index2):
        _, index = self.find_edge_in_digraph(index1, index2)
        return index >= 0

    def delete_edge(self, idx1, idx2):
        _, index = self.find_edge(idx1, idx2)
        if index >= 0:
            del self.edges[index]
            return True
        return False

    def delete_all(self):
        self.nodes = []
        self.edges = []
        self.weight = []

    ############################# PROJECT1 ################################

    ##### ex 1 #####
    def fill_from_adjacency_list(self, filename, digraph=False):
        with open(filename, 'r') as f:
            lines = f.readlines()
        rows = len(lines)
        self.add_nodes(rows)
        for i in range(rows):
            line = lines[i].split(' ')
            for j in range(len(line)):
                self.add_neighbour(i, int(line[j]) - 1)
                if(digraph):
                    self.add_edge_to_digraph(i, int(line[j]) - 1)    
                else:
                    self.add_edge(i, int(line[j]) - 1)

    def fill_from_adjacency_matrix(self, filename, digraph=False):
        with open(filename, 'r') as f:
            lines = f.readlines()
        rows = len(lines)
        self.add_nodes(rows)
        for i in range(rows):
            line = lines[i].split(' ')
            for j in range(len(line)):
                if int(line[j]) == 1:
                    self.add_neighbour(i, j)
                    if(digraph):
                        self.add_edge_to_digraph(i, j)
                    else:
                        if j < i:
                            self.add_edge(i, j)

    def fill_from_incidence_matrix(self, filename, digraph=False):
        with open(filename, 'r') as f:
            lines = f.readlines()
        line = lines[0].split(' ')
        cols = len(line)
        self.add_nodes(cols)
        for i in range(len(lines)):
            line = lines[i].split(' ')
            ones = 0
            for j in range(len(line)):
                if int(line[j]) == 1 and ones == 0:
                    start = j
                    ones += 1
                elif int(line[j]) == 1 and ones == 1:
                    end = j
                    ones += 1
            self.add_neighbours(start, end)
            if(digraph):
                self.add_edge_to_digraph(start, end)
            else:
                self.add_edge(start, end)

    def print_adjacency_list(self):
        print('Lista sąsiedztwa:')
        for node in self.nodes:
            node.print_neighbours_list()

    def print_adjacency_matrix(self):
        print('Macierz sąsiedztwa:')
        for node in self.nodes:
            node.print_neighbours_vector(len(self.nodes))

    def print_incidence_matrix(self):
        print('Macierz incydencji:')
        for edge in self.edges:
            edge.print_nodes_vector(len(self.nodes))

    def print_all_representations(self):
        self.print_adjacency_list()
        self.print_adjacency_matrix()
        self.print_incidence_matrix()

    ##### ex 2 #####
    def create_nx_graph(self):
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node.number + 1)
        for edge in self.edges:
            G.add_edge(edge.start + 1, edge.end + 1)
        return G

    def draw_nx_graph(self):
        G = self.create_nx_graph()
        nx.draw_circular(G , with_labels=True, font_weight='bold')
        plt.show()

    ##### ex 3 #####
    def fill_random_NL(self, n, l):
        if l > (n * (n - 1)) / 2:
            return False
        self.add_nodes(n)
        while len(self.edges) < l:
            index1 = random.randint(0, n - 1)
            index2 = random.randint(0, n - 1)
            self.add_edge(index1, index2)
            self.add_neighbours(index1, index2)
        return True

    def fill_random_NP(self, n, p):
        self.add_nodes(n)
        for i in self.nodes:
            for j in self.nodes:
                probability = random.random()
                if probability >= p:
                    self.add_edge(i.number, j.number)
                    self.add_neighbours(i.number, j.number)

    ############################# PROJECT2 ################################

    ##### ex 1 #####
    def fill_from_graphic_sequence(self, sequence):
        if not check_graphic_sequence(sequence):
            return False
        seq = [[idx, deg] for idx, deg in enumerate(sequence)]
        self.add_nodes(len(sequence))

        while True:
            seq.sort(reverse=True, key=itemgetter(1))
            if all(el[1] == 0 for el in seq):
                break
            for i in range(1, seq[0][1] + 1):
                seq[i][1] -= 1
                self.add_edge(seq[0][0], seq[i][0])
                self.add_neighbours(seq[0][0], seq[i][0])
            seq[0][1] = 0

    ##### ex 2 #####
    def swap_edges(self, edge1, edge2):
        a = edge1.start
        b = edge1.end
        c = edge2.start
        d = edge2.end

        edge1.end = d
        edge2.end = b
        self.nodes[a].delete_neighbour(b)
        self.nodes[b].delete_neighbour(a)
        self.nodes[c].delete_neighbour(d)
        self.nodes[d].delete_neighbour(c)

        self.add_neighbours(a, d)
        self.add_neighbours(c, b)

    def randomize_edges(self, n, seq):
        self.fill_from_graphic_sequence(seq)
        for _ in range(n):
            while True:
                idx1 = random.randint(0, len(self.edges) - 1)
                idx2 = random.randint(0, len(self.edges) - 1)
                if idx1 != idx2:
                    break
            self.swap_edges(self.edges[idx1], self.edges[idx2])

    ##### ex 3 #####
    def components_R(self, nr, v, comp):
        for u in self.nodes[v].neighbours:
            if comp[u] == -1:
                comp[u] = nr
                self.components_R(nr, u, comp)

    def components(self):
        nr = 0
        comp = [-1 for _ in range(len(self.nodes))]
        for v in range(len(self.nodes)):
            if comp[v] == -1:
                nr += 1
                comp[v] = nr
                self.components_R(nr, v, comp)
        return comp

    def largest_consistent_component(self):
        comp = list(self.components())
        comp.sort()
        largest = -1
        max_count = 0
        prev = -1
        for c in comp:
            if c != prev:
                prev = c
                count = 0
            count += 1
            if count > max_count:
                max_count = count
                largest = c
        return max_count

    ##### ex 4 #####
    def generate_even_sequence(self, n):
        seq = []
        for _ in range(n):
            a = 1
            b = int(n * (n - 1) / 4)
            value = random.randint(a, b) * 2

            seq.append(value)
        return seq

    def is_coherent(self):
        max_count = self.largest_consistent_component()
        return max_count == len(self.nodes)

    def fill_random_euler(self, n):
        seq = self.generate_even_sequence(n)
        self.fill_from_graphic_sequence(seq)
        while True:
            is_coherent = self.is_coherent()
            is_graphic = check_graphic_sequence(seq)
            if is_coherent and is_graphic:
                break
            seq = self.generate_even_sequence(n)
            self.fill_from_graphic_sequence(seq)

    def create_copy(self):
        copy = Graph()
        copy.nodes = [n for n in self.nodes]
        copy.edges = [e for e in self.edges]
        copy.weight = [w for w in self.weight]
        return copy

    def is_bridge(self, start, end):
        self.delete_edge(start, end)
        res = not self.is_coherent()
        self.add_edge(start, end)
        return res

    def find_euler_cycle(self):
        copy = self.create_copy()
        start_index = random.randint(0, len(copy.nodes) - 1)
        current_node = Node.create_copy(copy.nodes[start_index])
        # current_node = copy.nodes[start_index].create_copy()
        euler_cycle = [current_node.number]
        while True:
            edges_len = len(current_node.neighbours)
            end = edges_len == 0
            if end:
                break
            for neighbour_idx in current_node.neighbours:
                is_bridge = copy.is_bridge(current_node.number, neighbour_idx)
                edges_left = len(copy.edges)

                if not is_bridge:
                    copy.delete_edge(current_node.number, neighbour_idx)
                    copy.nodes[current_node.number].delete_neighbour(
                        neighbour_idx)
                    copy.nodes[neighbour_idx].delete_neighbour(
                        current_node.number)
                    current_node = copy.nodes[neighbour_idx]
                    euler_cycle.append(current_node.number)
                elif len(current_node.neighbours) == 1:
                    # copy.delete_node(current_node.num)
                    copy.delete_edge(current_node.number, neighbour_idx)
                    copy.nodes[current_node.number].delete_neighbour(
                        neighbour_idx)
                    copy.nodes[neighbour_idx].delete_neighbour(
                        current_node.number)
                    current_node = copy.nodes[neighbour_idx]
                    euler_cycle.append(current_node.number)

        return euler_cycle

    ##### ex 5 #####
    def fill_k_regular(self, n, k):
        seq = [k for _ in range(n)]
        self.fill_from_graphic_sequence(seq)

    ##### ex 6 #####
    def find_hamilton_cycle(self):
        coherent = self.is_coherent()
        if not coherent:
            print("Graf nie jest spójny")
            return False
        visited = [False for _ in range(len(self.nodes))]
        stack = []
        self.hamilton_R(0, visited, stack)
        if len(stack) < len(self.nodes):
            print("Nie znaleziono cyklu Hamiltona")

    def hamilton_R(self, v, visited, stack):
        stack.append(v)
        if len(stack) < len(self.nodes):
            visited[v] = True
            for u in self.nodes[v].neighbours:
                if visited[u] == False:
                    cycle = self.hamilton_R(u, visited, stack)
                    if cycle:
                        return cycle
            visited[v] = False
            stack.pop()
        else:
            test = False
            for u in self.nodes[v].neighbours:
                if u == 0:
                    test = True
            cycle = stack
            if test == True:
                print("Cykl Hamiltona:")
                cycle.append(0)
            else:
                print("Ścieżka Hamiltona:")
            print([number + 1 for number in stack])
            return cycle
        return None

    ############################# PROJECT3 ################################

    ##### ex 1 #####

    def generate_random_graph(self, min_nodes, max_nodes):
        # randomizes the number of nodes from the range
        # [min_nodes, max_nodes]
        nodes = random.randint(min_nodes, max_nodes)

        # min number of edges for coherent graph is n-1, max is n(n-1)/2
        edges = random.randint(nodes - 1, int(nodes * (nodes - 1) / 2))
        self.fill_random_NL(nodes, edges)

    def generate_random_coherent_graph(self, min_nodes, max_nodes):

        self.generate_random_graph(min_nodes, max_nodes)
        while not self.is_coherent():
            self.delete_all()
            self.generate_random_graph(min_nodes, max_nodes)

    def add_random_weight(self):
        self.weight = [random.randint(-5, 10) for _ in range(len(self.edges))]

    def draw_nx_graph_with_weight(self):
        G = nx.DiGraph()
        for node in self.nodes:
            G.add_node(node.number)
        for edge, weight in zip(self.edges, self.weight):
            G.add_edge(edge.start, edge.end, weight=weight)
        pos = nx.spring_layout(G)
        nx.draw_networkx(G, pos, with_labels=True, arrows=True, font_weight='bold')
        labels = nx.get_edge_attributes(G, 'weight')
        nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
        plt.show()

    ##### ex 2 #####

    def init_d_p(self, d, p, s):
        for i in range(len(self.nodes)):
            d.append(float('inf'))
            p.append(None)
        d[s] = 0

    def relax(self, u, neighbour, d, p):  # ( take in eeeeeesy LOVE MIKA)
        edges = [(edge.start, edge.end) for edge in self.edges]
        try:
            index = edges.index((u, neighbour))
        except ValueError:
            index = edges.index((neighbour, u))
        weight = self.weight[index]
        if d[neighbour] > (d[u] + weight):
            d[neighbour] = (d[u] + weight)
            p[neighbour] = u

    def dijkstra(self, p, d, s, print_s=True):

        self.init_d_p(d, p, s)
        S = []  # 'ready' nodes
        while len(S) != len(self.nodes):
            u = find_node_with_smallest_d(S, d)
            S.append(u)
            for neighbour in self.nodes[u].neighbours:
                if neighbour not in S:
                    self.relax(u, neighbour, d, p)

        if print_s:
            print_S(S, d, p)

    ##### ex 3 #####

    def distance_matrix(self, matrix):
        for i in range(len(self.nodes)):
            distance = []
            predecessor = []
            self.dijkstra(predecessor, distance, i, False)
            matrix[i] = distance
        
    ############################# PROJECT4 ################################
    def generate_random_digraph(self, min_nodes, max_nodes, p):
        nodes = random.randint(min_nodes, max_nodes)
        self.fill_random_NP_digraph(nodes, p)

    def fill_random_NP_digraph(self, n, p):
        self.add_nodes(n)
        for i in self.nodes:
            for j in self.nodes:
                print(str(i.number) + " " + str(j.number))
                probability = random.random()
                if probability <= p and i != j:
                    self.add_edge_to_digraph(i.number, j.number)
                    self.add_neighbour(i.number, j.number)

    def to_adjacency_matrix(self):
        nodes_len = len(self.nodes)
        zeros = [[0 for _ in range(nodes_len)] for _ in range(nodes_len)] 

        for node in self.nodes:
            for neighbour in node.neighbours:
                zeros[node.number][neighbour] = 1
        return zeros
    
    def from_adjacency_matrix(self, matrix):
        graph = Graph()
        graph.add_nodes(len(matrix))
        for row_idx in range(len(matrix)):
            row = matrix[row_idx]
            for col_idx in range(len(row)):
                if(matrix[row_idx][col_idx] > 0):
                    graph.add_neighbour(row_idx, col_idx)
                    graph.add_edge(row_idx, col_idx)
        return graph

    def transpose(self):
        matrix = self.to_adjacency_matrix()
        rows = len(matrix)
        columns = len(matrix[0])

        matrix_T = [[matrix[j][i] for j in range(rows)] for i in range(columns)]

        G_T = self.from_adjacency_matrix(matrix_T)
        return G_T
        

    def components_R2(self, v, visited, stack):
        for u in v.neighbours:
            if u not in visited:
                visited[u] = v
                self.components_R2(self.nodes[u], visited, stack)
        stack.append(v.number)

    ##### ex 1 #####

    def kosaraju(self):
        d = {node.number: -1 for node in self.nodes}
        stack = []
        t = 0

        for node in self.nodes:
            if d[node.number] == -1:
                self.dfs_visit_digraph(node, d, stack, t)

        G_T = self.transpose()
        components = []
        visited = {}
        i = 0
        while stack != []:
            v = stack.pop()
            if v.number in visited:
                continue
            else:
                components.append([])
                if v.number not in visited:
                    visited[v.number] = True
                    G_T.components_R2(v, visited, components[i])            
                i += 1

        return components

    
    
    def dfs_visit_digraph(self, v, d, stack, t):
        t += 1
        d[v.number] = t
        for neighbour in v.neighbours:
            if d[neighbour] == -1:
                self.dfs_visit_digraph(self.nodes[neighbour], d, stack, t)
        t += 1
        stack.append(v)

    ##### ex 2 #####

    def bellman_ford(self, src):
        dist = [float("Inf")] * len(self.nodes)
        dist[src] = 0
        p = [0] * len(self.nodes)

        for _ in range(len(self.nodes) - 1):
            for e in self.edges:
                self.relax(e.start, e.end, dist, p)

        for i, e in enumerate(self.edges):
            if dist[e.start] != float("Inf") and dist[e.start] + self.weight[i] < dist[e.end]:
                print("Graph contains negative weight cycle")
                return False
        return dist

    ##### ex 3 #####
    def add_S(self):
        G_ = self.create_copy()
        new_node = Node(len(self.nodes))
        G_.add_node(new_node)
        for node in G_.nodes:
            if node.number != new_node.number:
                G_.add_neighbour(new_node.number, node.number)
                G_.add_edge(new_node.number, node.number)
                G_.weight.append(0)
        
        return G_, new_node




    def johnson(self):
        G_, start = self.add_S()
        bell_dist = G_.bellman_ford(start.number)
        p = [0] * len(self.nodes)

        
        for node in G_.nodes:
            for neighbour in node.neighbours:
                w = G_.weight[neighbour]
                G_.weight[node.number] = w + bell_dist[node.number] - bell_dist[neighbour]

        G_.delete_node(start.number)
        dij_distance = [[] for _ in self.nodes]
        result = {}
        for u in G_.nodes:
            result[u.number] = {}
            G_.dijkstra(p, dij_distance[u.number], u.number, print_s=False)
            for v in self.nodes:
                result[u.number][v.number] = dij_distance[u.number][v.number] - bell_dist[u.number] + bell_dist[v.number]

        return result

    ###################### PROJECT 6 ###########################

    def generate_random_connected_digraph(self, min_nodes, max_nodes, p):
        n = random.randint(min_nodes, max_nodes)
        self.add_nodes(n)
        for i in range(len(self.nodes)):
            self.add_edge_to_digraph(i, (i+1)%n)
            self.add_neighbour(i, (i+1)%n)
        for i in self.nodes:
            for j in self.nodes:
                probability = random.random()
                if probability <= p and i != j and not self.edge_exists_in_digraph(i, j):
                    self.add_edge_to_digraph(i.number, j.number)
                    self.add_neighbour(i.number, j.number)

    def probability_page_rank(self, d=0.15, N=100_000):
        result = [0 for _ in range(len(self.nodes))]
        current = random.randint(0, len(self.nodes)-1)
        for _ in range(N):
            p = random.random()
            if p < d:
                current = random.randint(0, len(self.nodes)-1)
            else:
                neighbours = self.nodes[current].neighbours
                current = neighbours[random.randint(0, len(neighbours)-1) if len(neighbours)>1 else 0]
            result[current] += 1
        result = list(map(lambda x: x/N, result))
        print(f"Prob: {result}")
        
    def iterative_page_rank(self, d=0.15, N=100_000):
        pt = np.array([1/len(self.nodes) for _ in range(len(self.nodes))])
        A = np.zeros((len(self.nodes), len(self.nodes)))
        for i in self.nodes:
            for j in i.neighbours:
                A[i.number][j] = 1/len(i.neighbours)
        from_adjacency_matrix = lambda x: (1-d)*x + d/(len(self.nodes))
        P = np.transpose(from_adjacency_matrix(A))
        for _ in range(N):
           pt = P.dot(pt)
        print(f"Iter: {list(map(lambda x: round(x,5), pt.tolist()))}")


        

    













