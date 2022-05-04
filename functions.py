
def check_graphic_sequence(seq):
    seq = list(seq)
    seq.sort(reverse=True)
    while True:
        if all(number == 0 for number in seq):
            return True
        if any(number < 0 for number in seq) or seq[0] >= len(seq):
            return False
        for i in range(1,seq[0]+1):
            seq[i] -= 1
        seq[0] = 0
        seq.sort(reverse=True)

