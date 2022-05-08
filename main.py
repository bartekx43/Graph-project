from nis import match
from select import select
from tkinter import (filedialog, simpledialog, messagebox)
from graph import *
from functions import check_graphic_sequence
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Graphs Theory')
    parser.add_argument('--project', type=int, default=None, help='Project number')
    parser.add_argument('--ex', type=int, default=None, help='Exercise number')
    parser.add_argument('--visual', type=bool, default=False, help='Draw graph')
    args = parser.parse_args()

    if args.project is None:
        print("brak flagi --project")
        exit(-1)
    if args.ex is None:
        print("brak flagi --ex")
        exit(-1)

    # filepath = filedialog.askopenfilename(initialdir='examples', filetypes=(("Text files", "*.txt"), ("all files", "*.*")))
    graph = Graph()
    seq_graphic = [ 4, 2, 2, 3, 2, 1, 4, 2, 2, 2, 2]

    # required python 3.10
    match args.project:
        case 1:
            match args.ex:
                case 1:
                    print('---Graf z listy sąsiedztwa---')
                    graph.fill_from_adjacency_list("./files/AL.txt")
                    graph.print_all_representations()
                    graph.delete_all()
                    print('---Graf z macierzy sąsiedztwa---')
                    graph.fill_from_adjacency_matrix('./files/AM.txt')
                    graph.print_all_representations()
                    graph.delete_all()
                    print('---Graf z macierzy incydencji---')
                    graph.fill_from_incidence_matrix('./files/IM.txt')
                case 2:
                    graph.fill_random_NP(5, 0.8)
                    graph.draw_nx_graph()
                case 3:
                    graph.fill_random_NL(5,3)
                    print("---Graf losowy(n,l)---")
                    graph.print_all_representations()
                    graph.delete_all()
                    graph.fill_random_NP(5, 0.8)
                    print("---Graf losowy(n,p)---")
                case _:
                    print("Nie ma takiego zadania")
        case 2:
            match args.ex:
                case 1:
                    graph.fill_from_graphic_sequence(seq_graphic)
                    print(f'Ciąg graficzny: {seq_graphic}')
                case 2:
                    graph.randomize_edges(10, seq_graphic)
                case 3:
                    graph.fill_from_graphic_sequence(seq_graphic)
                    count = graph.largest_consistent_component()
                    print(f"Wielkość największej spójnej składowej: {count}")
                case 4:
                    graph.fill_random_euler(5)
                    cycle = graph.find_euler_cycle()
                    print(f'Cykl Eulera: {cycle}')
                case 5:
                    graph.fill_k_regular(5, 2)
                case 6:
                    # graph.randomize_edges(10, seq_graphic) # z tego raczej nie będzie cyklu hamiltona
                    # graph.fill_from_graphic_sequence([3,4,3,4,3,3,3,3]) # z tego na pewno będzie cykl hamiltona
                    graph.fill_k_regular(6, 3)
                    graph.find_hamilton_cycle()
                case _:
                    print("Nie ma takiego zadania")
        case _:
            print("Nie ma takiego projektu")
    graph.print_all_representations()
    if args.visual:
        graph.draw_nx_graph()
    graph.delete_all()