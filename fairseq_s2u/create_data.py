import argparse
import pandas as pd
import csv
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--tsv', required=True)
    parser.add_argument('--audio-key', default='audio')
    parser.add_argument('--text-key', default='tgt_text')
    parser.add_argument('--lang', default='en')
    parser.add_argument('--quant-dir', required=True)
    parser.add_argument('--quant-lang', default='quant')
    parser.add_argument('--output-prefix', required=True)
    args = parser.parse_args()

    clust_data = {}

    for file in Path(args.quant_dir).iterdir():

        if file.is_file():

            with open(file) as f:
                for line in f:
                    audio, seq = line.strip().split('|')
                    assert audio not in clust_data
                    clust_data[audio] = seq

    data = pd.read_csv(
        args.tsv,
        sep = '\t',
        quoting = csv.QUOTE_NONE,
        doublequote = False,
        quotechar=None,
    )
    
    with open(f'{args.output_prefix}.{args.lang}', 'w') as text_f:
        with open(f'{args.output_prefix}.{args.quant_lang}', 'w') as quant_f:
            for idx, sample in data.iterrows():
                text = sample[args.text_key]
                audio = sample[args.audio_key].rsplit('.', maxsplit=1)[0]
                assert audio in clust_data, f'audio: {audio}, args: {args}'
                seq = clust_data[audio]
                # text = text.replace(' ', '_')
                # print("_ "+" ".join(list(text)), file=text_f)
                print(text, file=text_f)
                print(seq, file=quant_f)


        