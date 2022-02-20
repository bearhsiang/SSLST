import argparse
from pathlib import Path

def simple(s: str) -> str:

    l = s.split()
    reduced_l = []
    for c in l:
        if len(reduced_l) == 0 or reduced_l[-1] != c:
            reduced_l.append(c)
    reduced_s = ' '.join(reduced_l)
    
    return reduced_s

def addN(s):

    l = s.split()
    reduced_l = [l[0]]
    l = l[1:]

    count = 1
    for c in l:
        if reduced_l[-1] == c:
            count += 1
        else:
            reduced_l.append(f'_{count}')
            reduced_l.append(c)
            count = 1
    reduced_l.append(f'_{count}')
    reduced_s = ' '.join(reduced_l)

    return reduced_s

reducers = {
    'simple': simple,
    'addN': addN,
}

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('-m', '--mode', default='simple', choices=reducers.keys())
    args = parser.parse_args()
    return args

def main(args):

    reducer = reducers[args.mode]
    output_file = Path(args.output)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f_out:
        with open(args.input, 'r') as f_in:

            for line in f_in:
                s = line.strip()
                s = reducer(s)
                print(s, file=f_out)

if __name__ == '__main__':

    args = get_args()
    main(args)