import argparse
from datasets import datasets
from pathlib import Path

import pandas as pd
import csv

def get_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--dataset', choices=datasets.keys())
    parser.add_argument('-r', '--root-dir')
    parser.add_argument('-v', '--split', help = "process all the splits if not set.")
    parser.add_argument('-s', '--src-lang')
    parser.add_argument('-t', '--tgt-lang')
    parser.add_argument('-o', '--output-dir', default='data/tsv')
    
    args = parser.parse_args()
    
    return args

def main():

    args = get_args()
    dataset_cls = datasets[args.dataset]
    splits = [args.split] if args.split else dataset_cls.get_splits()

    output_dir = Path(args.output_dir) / f'{dataset_cls.get_name()}-{args.src_lang}-{args.tgt_lang}'
    output_dir.mkdir(parents=True, exist_ok=True)

    for split in splits:
        
        dataset = dataset_cls(
            Path(args.root_dir),
            split,
            args.src_lang,
            args.tgt_lang,
        )

        data = [item for item in dataset]
        df = pd.DataFrame(data)

        df.to_csv(
            output_dir/f'{split}.tsv', 
            sep='\t',
            index=False,
            quoting=csv.QUOTE_NONE,
        )

if __name__ == "__main__":
    main()