import argparse

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input')
    parser.add_argument('--mapping')
    parser.add_argument('--output')
    args = parser.parse_args()

    return args

def main(args):

    mapping = {}
    for line in open(args.mapping):
        src, tgt = line.strip().split()
        mapping[src] = tgt
    
    with open(args.output, 'w') as f:
        for line in open(args.input):
            s = ' '.join([mapping[w] for w in line.strip().split()])
            print(s, file=f)
    
if __name__ == '__main__':

    args = get_args()
    main(args)