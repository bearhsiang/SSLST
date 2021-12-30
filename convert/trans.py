import argparse
import pandas as pd
import csv
import sentencepiece as spm

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-tsv', required=True)
    parser.add_argument('--output-prefix', required=True)
    parser.add_argument('--src-key', required=True)
    parser.add_argument('--src-lang', required=True)
    parser.add_argument('--src-bpe')

    parser.add_argument('--tgt-key', required=True)
    parser.add_argument('--tgt-lang', required=True)
    parser.add_argument('--tgt-bpe')
    args = parser.parse_args()
    return args

def apply_bpe_and_dump(df, key, bpe_model, output):

    texts = df[key]
    if bpe_model:
        model = spm.SentencePieceProcessor(model_file=bpe_model)
        def fn(s):
            l = model.encode(s, out_type=str)
            return ' '.join(l)
        texts = texts.map(fn)

    with open(output, 'w') as f:
        for text in texts:
            print(text, file=f)

def main(args):

    data = pd.read_csv(
        args.input_tsv,
        delimiter='\t',
        quoting=csv.QUOTE_NONE,
    )

    apply_bpe_and_dump(data, args.src_key, args.src_bpe, f'{args.output_prefix}.{args.src_lang}')
    apply_bpe_and_dump(data, args.tgt_key, args.tgt_bpe, f'{args.output_prefix}.{args.tgt_lang}')



if __name__ == '__main__':
    args = get_args()
    main(args)