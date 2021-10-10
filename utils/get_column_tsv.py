from data_utils import read_tsv
import argparse
from tqdm.auto import tqdm

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-tsv', required=True)
    parser.add_argument('-k', '--key', required=True)
    args = parser.parse_args()

    lines = read_tsv(args.input_tsv)
    for line in tqdm(lines):
        assert args.key in line
        print(line[args.key])
    