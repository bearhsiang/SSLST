import argparse
import pandas as pd
from pathlib import Path
import csv

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-k', '--key', required=True)
    parser.add_argument('-o', '--output', required=True)
    parser.add_argument('--na-rep', default="[REMOVE]")
    args = parser.parse_args()
    
    return args

def main(args):

    df = pd.read_csv(args.input, 
        sep= '\t' if Path(args.input).suffix == '.tsv' else ',',
        quoting=csv.QUOTE_NONE,
    )
    assert args.key in df, f'key "{args.key}" is not exist in {args.input}'
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    df[args.key].to_csv(
        args.output,
        index=False,
        header=False,
        sep='\t',
        quoting=csv.QUOTE_NONE,
        na_rep=args.na_rep,
    )

if __name__ == '__main__':

    args = get_args()
    main(args)