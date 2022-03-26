import argparse
import numpy as np

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix')
    parser.add_argument('--src-lang')
    parser.add_argument('--tgt-lang')
    args = parser.parse_args()

    return args

def main(args):

    src_data_file = f'{args.prefix}.{args.src_lang}'
    tgt_data_file = f'{args.prefix}.{args.tgt_lang}'

    f_src = open(src_data_file)
    f_tgt = open(tgt_data_file)

    results = {
        'src_len': [],
        'tgt_len': [],
        'ratio': [],
    }

    for src, tgt in zip(f_src, f_tgt):

        src = src.strip().split()
        tgt = tgt.strip().split()

        results['src_len'].append(len(src))
        results['tgt_len'].append(len(tgt))
        results['ratio'].append(len(tgt)/len(src))
    
    for k in results:
        print(k)
        print(f'\t25%: {np.percentile(results[k], 25):.2f}')
        print(f'\t50%: {np.percentile(results[k], 50):.2f}')
        print(f'\t75%: {np.percentile(results[k], 75):.2f}')
        print(f'\tave: {np.mean(results[k]):.2f}')
        print(f'\tmin: {min(results[k])}')
        print(f'\tmax: {max(results[k])}')

if __name__ == '__main__':

    args = get_args()
    main(args)