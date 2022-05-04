
class Edge:
    def __init__(self, start, end):
        self.start = start
        self.end = end

    def print_nodes_vector(self, total):
        vector = [0 for _ in range(total)]
        vector[self.start] = 1
        vector[self.end] = 1
        print(vector)