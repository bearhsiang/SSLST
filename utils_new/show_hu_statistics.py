import argparse
from pathlib import Path
import matplotlib.pyplot as plt

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dir', required=True)
    parser.add_argument('-s', '--splits', default='train,dev,test')
    parser.add_argument('-l', '--lang', required=True)
    args = parser.parse_args()

    return args

def main(args):

    record = {}

    data_dir = Path(args.dir)

    for split in args.splits.split(','):
        for line in open(data_dir/f'{split}.{args.lang}'):
            line = line.strip().split()
            for w in line:
                if w in record:
                    record[w] += 1
                else:
                    record[w] = 1
    
    w_count = [ i[1] for i in record.items() ]
    w_count.sort(reverse=True)

    plt.bar(list(range(len(w_count))), w_count)
    plt.savefig('test.png')


if __name__ == '__main__':

    args = get_args()
    main(args)