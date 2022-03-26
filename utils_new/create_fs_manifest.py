import argparse
import pandas as pd
import csv
from pathlib import Path
import torchaudio
from tqdm.auto import tqdm

def get_wav(audio_root, audio_list):
    for audio_name in tqdm(audio_list):
        wav, sr = torchaudio.load(audio_root/audio_name)
        yield audio_name, wav.mean(0), sr

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--audio-root', required=True)
    parser.add_argument('--tsv-file', required=True)
    parser.add_argument('--key', default='audio')
    parser.add_argument('--output', default='manifest.txt')
    parser.add_argument('--sample-rate', default=16000, type=int)
    args = parser.parse_args()

    data = pd.read_csv(
        args.tsv_file,
        delimiter='\t',
        quoting=csv.QUOTE_NONE,
    )

    parent_dir = Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    audio_root = Path(args.audio_root)

    with open(args.output, 'w') as f:
        print(args.audio_root, file=f)
        for audio_name, wav, sr in get_wav(audio_root, list(data[args.key])):
            n_frame = int(len(wav) * args.sample_rate / sr)
            print(audio_name, n_frame, sep='\t', file=f)
