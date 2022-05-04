
class Node:
    def __init__(self, number):
        self.number = number
        self.neighbours = []
    
    def print_neighbours_list(self):
        print("{}: {}".format(self.number+1, [n+1 for n in self.neighbours]))

    def print_neighbours_vector(self, total):
        vector = [0 for _ in range(total)]
        for i in range(len(self.neighbours)):
            vector[self.neighbours[i]] = 1
        print(vector)