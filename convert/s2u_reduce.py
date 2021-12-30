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

reducers = {
    'simple': simple,
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
                index, s = line.strip().split('|')
                s = reducer(s)
                print(index, s, sep='|', file=f_out)

if __name__ == '__main__':

    args = get_args()
    main(args)