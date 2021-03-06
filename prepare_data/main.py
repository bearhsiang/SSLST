import argparse
from datasets import datasets
from pathlib import Path
import pandas as pd
import csv
from tqdm.auto import tqdm

def get_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-d', '--dataset', choices=datasets.keys(), required=True)
    parser.add_argument('-r', '--root-dir', required=True)
    parser.add_argument('-v', '--split', help = "process all the splits if not set.")
    parser.add_argument('-s', '--src-lang')
    parser.add_argument('-t', '--tgt-lang')
    parser.add_argument('-o', '--output-root', default='data/tsv')
    
    args = parser.parse_args()
    
    return args

def main():

    args = get_args()
    dataset_cls = datasets[args.dataset]
    splits = [args.split] if args.split else dataset_cls.get_splits()

    for split in splits:
        
        dataset = dataset_cls(
            Path(args.root_dir),
            split,
            args.src_lang,
            args.tgt_lang,
        )

        output_dir = Path(args.output_root) / dataset.name()
        output_dir.mkdir(parents=True, exist_ok=True)

        data = [item for item in tqdm(dataset)]
        df = pd.DataFrame(data)

        df.to_csv(
            output_dir/f'{split}.tsv', 
            sep='\t',
            index=False,
            quoting=csv.QUOTE_NONE,
        )

if __name__ == "__main__":
    main()