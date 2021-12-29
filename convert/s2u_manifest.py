import argparse
import pandas as pd
import csv
from pathlib import Path

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-root', required=True)
    parser.add_argument('--tsv-file', required=True)
    parser.add_argument('--key', default='audio')
    parser.add_argument('--output', default='manifest.txt')
    args = parser.parse_args()

    data = pd.read_csv(
        args.tsv_file,
        delimiter='\t',
        quoting=csv.QUOTE_NONE,
    )

    parent_dir = Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    with open(args.output, 'w') as f:
        print(args.audio_root, file=f)
        for audio_name in data[args.key]:
            print(audio_name, -1, sep='\t', file=f)
