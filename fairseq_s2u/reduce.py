import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    args = parser.parse_args()

    with open(args.input, 'r') as in_f:
        with open(args.output, 'w') as out_f:
            for line in in_f:
                line = line.strip().split()
                l = []
                for i in line:
                    if len(l) == 0 or l[-1] != i:
                        l.append(i)
                print(" ".join(l), file=out_f)

