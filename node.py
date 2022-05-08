
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

    def delete_neighbour(self, index):
        for i, n in enumerate(self.neighbours):
            if n == index:
                del self.neighbours[i]
                break
    
    @staticmethod
    def create_copy(node):
        copy = Node(node.number)
        copy.neighbours = [n for n in node.neighbours]
        return copy