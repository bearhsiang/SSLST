import argparse
from collections import defaultdict

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    args = parser.parse_args()
    
    return args

def main(args):

    record = defaultdict(int)

    with open(args.input) as f:
        for line in f:
            line = line.strip().split('|')[-1]
            line = line.split()
            for w in line:
                record[w] += 1
    
    print(record)
            

if __name__ == '__main__':

    args = get_args()
    main(args)