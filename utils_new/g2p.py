import argparse
import g2p_en
from tqdm.auto import tqdm

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input')
    parser.add_argument('-o', '--output')
    parser.add_argument('-l', '--lang')
    args = parser.parse_args()

    return args

def main(args):

    model = g2p_en.G2p()

    with open(args.output, 'w') as f:
        for line in tqdm(open(args.input)):
            line = line.strip()
            p_list = model(line)
            while ' ' in p_list:
                p_list.remove(' ')
            print(*p_list, file=f)

if __name__ == '__main__':

    args = get_args()
    main(args)