import csv
import argparse
import string
from pathlib import Path
import pandas as pd
import sacremoses

def get_args():

    parser = argparse.ArgumentParser()
    
    parser.add_argument('-i', '--input_tsv', required=True)
    parser.add_argument('-o', '--output_tsv', required=True)
    parser.add_argument('-k', '--key', required=True)
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('-l', '--lowercase', action='store_true')
    parser.add_argument('-r', '--remove-punctuation', action='store_true')
    parser.add_argument('-c', '--remove-consecutive-blank', action='store_true')
    parser.add_argument('-L', '--lang', default='en')
    parser.add_argument('-f', '--force', action='store_true')
    
    args = parser.parse_args()

    return args

def main():

    args = get_args()

    output_tsv = Path(args.output_tsv)
    if not args.force:
        assert not output_tsv.exists(), f'output file: {output_tsv} exists, use -f/--force to force overwrite'

    output_tsv.parent.mkdir(parents=True, exist_ok=True)

    normalizer = sacremoses.MosesPunctNormalizer(
        lang=args.lang,
        pre_replace_unicode_punct=True,
        post_remove_control_chars=True,
    )

    data = pd.read_csv(
        args.input_tsv,
        sep='\t',
        quoting=csv.QUOTE_NONE
    )

    def process_fn(s):

        if args.normalize:
            s = normalizer.normalize(s)
        if args.remove_punctuation:
            for c in string.punctuation:
                s = s.replace(c, '')
        if args.lowercase:
            s = s.lower()
        if args.remove_consecutive_blank:
            s = ' '.join(s.split())
        
        return s

    data[args.key] = data[args.key].map(process_fn)
    data.to_csv(
        output_tsv,
        sep = '\t',
        quoting= csv.QUOTE_NONE,
        index=False,
    )

if __name__ == '__main__':

    main()