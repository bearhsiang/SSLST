import argparse
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm.auto import tqdm
import csv

UNK_TOKEN, UNK_TOKEN_ID = "<unk>", 3
BOS_TOKEN, BOS_TOKEN_ID = "<s>", 0
EOS_TOKEN, EOS_TOKEN_ID = "</s>", 2
PAD_TOKEN, PAD_TOKEN_ID = "<pad>", 1

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--in-dict')
    parser.add_argument('--out-dict')
    args = parser.parse_args()

    return args

def main(args):

    words = [line.split()[0] for line in open(args.in_dict)]

    for special_token in [UNK_TOKEN, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN]:
        if special_token in words:
            words.remove(special_token)

    with open(args.out_dict, 'w') as f:
        for w in words:
            print(w, 1, file=f)


if __name__ == '__main__':

    args = get_args()
    main(args)
