import random
from edge import *
from node import *

class Graph:
    def __init__(self, nodes=[], edges=[]):
        self.nodes = [n for n in nodes]
        self.edges = [e for e in edges]

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, start, end):
        if start == end or self.edge_exists(start, end):
            return False
        self.edges.append(Edge(start, end))
        return True

    def add_nodes(self, amount):
        for i in range(amount):
            self.add_node(Node(i))

    def add_neighbour(self, node_index, neighbour):
        if node_index == neighbour:
            return False
        self.nodes[node_index].neighbours.append(neighbour)
        return True

    def add_neighbours(self, node1, node2):
        if node1 == node2:
            return False
        self.nodes[node1].neighbours.append(node2)
        self.nodes[node2].neighbours.append(node1)
        return True

    def edge_exists(self, index1, index2):
        for edge in self.edges:
            if edge.start == index1 and edge.end == index2:
                return True
            elif edge.start == index2 and edge.end == index1:
                return True
        return False

    def fill_from_adjacency_list(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        rows = len(lines)
        self.add_nodes(rows)
        for i in range(rows):
            line = lines[i].split(' ')
            for j in range(len(line)):
                self.add_neighbour(i, int(line[j])-1)
                self.add_edge(i, int(line[j])-1)

    def fill_from_adjacency_matrix(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        rows = len(lines)
        self.add_nodes(rows)
        for i in range(rows):
            line = lines[i].split(' ')
            for j in range(len(line)):
                if int(line[j]) == 1:
                    self.add_neighbour(i, j)
                    if j < i:
                        self.add_edge(i, j)

    def fill_from_incidence_matrix(self, filename):
        with open(filename, 'r') as f:
            lines = f.readlines()
        line = lines[0].split(' ')
        cols = len(line)
        self.add_nodes(cols)
        for i in range(len(lines)):
            line = lines[i].split(' ')
            print(line)
            ones = 0
            for j in range(len(line)):
                if int(line[j]) == 1 and ones == 0:
                    start = j
                    ones += 1
                elif int(line[j]) == 1 and ones == 1:
                    end = j
                    ones += 1
            self.add_edge(start, end)
            print(start, end)
            self.add_neighbours(start, end)

    def print_adjacency_list(self):
        print('Adjacency list:')
        for node in self.nodes:
            node.print_neighbours_list()

    def print_adjacency_matrix(self):
        print('Adjacency matrix:')
        for node in self.nodes:
            node.print_neighbours_vector(len(self.nodes))

    def print_incidence_matrix(self):
        print('Incidence matrix:')
        for edge in self.edges:
            edge.print_nodes_vector(len(self.nodes))

    def fillRandomNL(self, n, l):
        if l > n-1:
            return False
        self.add_nodes(n)
        while len(self.edges) < l:
            index1 = random.randint(0,n-1)
            index2 = random.randint(0,n-1)
            self.add_edge(index1, index2)
            self.add_neighbours(index1, index2)
        return True

    def fillRandomNP(self, n, p):
        self.add_nodes(n)
        for i in self.nodes:
            for j in self.nodes:
                probability = random.uniform(0,1)
                if probability >= p:
                    self.add_edge(i.number, j.number)
                    self.add_neighbours(i.number, j.number)




