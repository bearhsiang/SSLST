import argparse

from pandas.io import parsers
from datasets import datasets
from pathlib import Path

import pandas as pd

def get_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--dataset', choices=datasets.keys())
    parser.add_argument('-r', '--root-dir')
    parser.add_argument('-v', '--split')
    parser.add_argument('-s', '--src-lang')
    parser.add_argument('-t', '--tgt-lang')
    parser.add_argument('-o', '--output-dir', default='data/tsv')
    
    args = parser.parse_args()
    
    return args

def main():

    args = get_args()
    
    dataset = datasets[args.dataset](
        args.root_dir,
        args.split,
        args.src_lang,
        args.tgt_lang,
    )

    data = [item for item in dataset]
    
    df = pd.DataFrame(data)

    # output
    output_dir = Path(args.output_dir) / dataset.__repr__()

    output_dir.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        output_dir/f'{args.split}.tsv', 
        sep='\t',
        index=False
    )

if __name__ == "__main__":
    main()