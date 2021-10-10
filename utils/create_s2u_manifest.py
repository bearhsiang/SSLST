import argparse
from data_utils import read_tsv

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-root', required=True)
    parser.add_argument('--tsv-file', required=True)
    parser.add_argument('--key', default='audio')
    parser.add_argument('--output-prefix', default='manifest')
    parser.add_argument('-N', type=int, default=-1)
    args = parser.parse_args()

    data = read_tsv(args.tsv_file)
    total = len(data)
    if args.N < 0:
        args.N = total
    for i in range(0, total, args.N):
        with open(f'{args.output_prefix}{i}.txt', 'w') as f:
            print(args.audio_root, file=f)
            for sample in data[i:i+args.N]:
                print(sample[args.key], -1, sep='\t', file=f)
    # with open(args.output, 'w') as f:
    #     print(args.audio_root, file=f)
    #     for sample in data:
    #         print(sample[args.key], -1, sep='\t', file=f)


