import argparse
import yaml
import os
import json

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--sp-model', required=True)
    parser.add_argument('-f', '--feat-file', required=True)
    parser.add_argument('-v', '--vocab-file', required=True)
    parser.add_argument('-a', '--audio-dir', required=True)
    parser.add_argument('-o', '--output', required=True)
    args = parser.parse_args()

    config = {
        'bpe_tokenizer': {
            'bpe': 'sentencepiece',
            'sentencepiece_model': args.sp_model,
        },
        'vocab_filename': args.vocab_file,
        'audio_root': args.audio_dir,
        'input_feat_per_channel': json.load(open(args.feat_file))
    }

    with open(args.output, 'w') as f:
        yaml.dump(config, f)
