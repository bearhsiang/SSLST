import csv
import argparse
import string
from tqdm.auto import tqdm
import os
from data_utils import read_tsv, write_tsv

def length(s):
    return len(s.split())

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_tsv', required=True)
    parser.add_argument('-o', '--output_tsv', required=True)
    parser.add_argument('-s', '--src-key', default='src_text')
    parser.add_argument('-t', '--tgt-key', default='tgt_text')
    parser.add_argument('-f', '--force', action='store_true')
    parser.add_argument('-u', '--max', type=int)
    parser.add_argument('-l', '--min', type=int)
    parser.add_argument('-r', '--ratio', type=float)
    parser.add_argument('-c', '--contain')
    args = parser.parse_args()

    if not args.force:
        assert not os.path.exists(args.output_tsv), f'output file: {args.output_tsv} exists, use -f/--force to force overwrite'
    
    if args.ratio:
        assert args.min >= 1, "minimum length should >= 1 with ratio test"
    
    lines = read_tsv(args.input_tsv)

    block_word_list = []
    if args.contain:
        block_word_list = args.contain.split(',')

    data = []
    for line in tqdm(lines):

        src_text = line[args.src_key]
        tgt_text = line[args.tgt_key]

        src_len = length(src_text)
        tgt_len = length(tgt_text)

        skip = False
        for block_word in block_word_list:
            if block_word in src_text or block_word in tgt_text:
                print(f"{line} contains word \"{block_word}\", skip")
                skip = True
                break
        if skip: 
            continue

        if args.max:
            if src_len > args.max or tgt_len > args.max:
                print(f"{line} text part too long, skip")
                continue
        
        if args.min:
            if src_len < args.min or tgt_len < args.min:
                print(f"{line} text part too short, skip")
                continue

        if args.ratio:
            if src_len/tgt_len > args.ratio or tgt_len/src_len > args.ratio:
                print(f"{line} text part invalid ratio, skip")
                continue

        data.append(line)

    print(f'remove {len(lines)-len(data)}/{len(lines)} samples')

    write_tsv(args.output_tsv, data)
