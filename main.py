from tkinter import (filedialog, simpledialog, messagebox)
from graph import *

if __name__=="__main__":
    # filepath = filedialog.askopenfilename(initialdir='examples', filetypes=(("Text files", "*.txt"), ("all files", "*.*")))
    graph = Graph()
    # graph.fill_from_adjacency_list(filepath)
    # graph.fill_from_adjacency_matrix(filepath)
    # graph.fill_from_incidence_matrix(filepath)
    # graph.fillRandomNL(5,3)
    graph.fillRandomNP(3, 0.5)
    graph.print_adjacency_list()
    graph.print_adjacency_matrix()
    graph.print_incidence_matrix()