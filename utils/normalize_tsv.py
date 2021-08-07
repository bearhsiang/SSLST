import csv
import argparse
import string
from tqdm.auto import tqdm
import os
import sacremoses
from data_utils import read_tsv, write_tsv

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_tsv', required=True)
    parser.add_argument('-o', '--output_tsv', required=True)
    parser.add_argument('-k', '--key', required=True)
    parser.add_argument('-n', '--normalize', action='store_true')
    parser.add_argument('-l', '--lowercase', action='store_true')
    parser.add_argument('-r', '--remove-punctuation', action='store_true')
    parser.add_argument('-L', '--lang', default='en')
    parser.add_argument('-f', '--force', action='store_true')
    args = parser.parse_args()
    
    if not args.force:
        assert not os.path.exists(args.output_tsv), f'output file: {args.output_tsv} exists, use -f/--force to force overwrite'

    normalizer = sacremoses.MosesPunctNormalizer(
        lang=args.lang,
        pre_replace_unicode_punct=True,
        post_remove_control_chars=True,
    )

    p_list = set(string.punctuation) - set("'-")

    lines = read_tsv(args.input_tsv)

    data = []

    for line in tqdm(lines):

        text = line[args.key]

        if args.normalize:
            text = normalizer.normalize(text)
            for c in ".?,'":
                text = text.replace(f' {c}', c)
            text = text.replace('do n\'t', 'don\'t')
        if args.remove_punctuation:
            text = ''.join([ c if c not in p_list else '' for c in text])
            text = ' '.join(text.split())
        if args.lowercase:
            text = text.lower()
        line[args.key] = text
        data.append(line)
    
    write_tsv(args.output_tsv, data)