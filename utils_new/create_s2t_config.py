import argparse
import yaml

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('--vocab-filename', type=str)
    parser.add_argument('--bpe-model', type=str)
    parser.add_argument('--audio-root', type=str, default='')
    parser.add_argument('--input-feat-per-channel', type=int)
    parser.add_argument('--use-audio-input', action='store_true', default=False)
    parser.add_argument('--output', required=True)
    parser.add_argument('--shuffle', action='store_true', default=False)
    args = parser.parse_args()

    return args

def main(args):

    config = {
        'vocab_filename': args.vocab_filename,
        'audio_root': args.audio_root,
        'bpe_tokenizer': {
            'bpe': 'sentencepiece',
            'sentencepiece_model': args.bpe_model,
        },
        'input_feat_per_channel': args.input_feat_per_channel,
        'use_audio_input': args.use_audio_input,
        'shuffle': args.shuffle,
    }

    yaml.dump(config, open(args.output, 'w'))
    

if __name__ == '__main__':

    args = get_args()
    main(args)
