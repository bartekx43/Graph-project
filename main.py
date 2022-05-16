from graph import *
import argparse

# PRZYKŁADOWE URUCHOMIENIE PROGRAMU: python3 main.py --project 1 --ex 1 --visual True
# W przypadku kilku grafów w jednym zadaniu rysuje się ostatni, by rysował się inny nalezy zakomentować wypełnianie następnych grafów

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

    graph = Graph()
    seq_graphic = [ 4, 2, 2, 3, 2, 1, 4, 2, 2, 2, 2]
    # seq_graphic = [ 3, 3, 4, 4 ,4, 3,3,4,4,1,1,2]

    if args.project == 1:
        if  args.ex == 1:
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
        if args.ex == 2:
            graph.fill_random_NP(5, 0.8)
            graph.draw_nx_graph()
        if  args.ex == 3:
            graph.fill_random_NL(8,28)
            print("---Graf losowy(n,l)---")
            graph.print_all_representations()
            graph.delete_all()
            graph.fill_random_NP(5, 1)
            print("---Graf losowy(n,p)---")
    if args.project == 2:
        if args.ex == 1:
            graph.fill_from_graphic_sequence(seq_graphic)
            print(f'Ciąg graficzny: {seq_graphic}')
        if args.ex == 2:
            graph.randomize_edges(10, seq_graphic)
        if args.ex == 3:
            graph.randomize_edges(10, seq_graphic)
            count = graph.largest_consistent_component()
            print(f"Wielkość największej spójnej składowej: {count}")
        if args.ex == 4:
            graph.fill_random_euler(8)
            cycle = graph.find_euler_cycle()
            print(f'Cykl Eulera: {cycle}')
        if args.ex == 5:
            graph.fill_k_regular(8, 6)
        if args.ex == 6:
            graph.fill_k_regular(10, 2)
            graph.find_hamilton_cycle()
    graph.print_all_representations()
    if args.visual:
        graph.draw_nx_graph()
    graph.delete_all()