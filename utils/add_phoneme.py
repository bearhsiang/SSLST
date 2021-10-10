from textgrid import TextGrid
import argparse
from data_utils import read_tsv, write_tsv
from tqdm.auto import tqdm
from pathlib import Path

def tg2labels(tg, rate):
    l = []
    for seg in tg:
        begin = int(seg.minTime/rate)
        end = int(seg.maxTime/rate)
        label = seg.mark if seg.mark != '' else 'sil'
        l += [label]*(end-begin)
    return l

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-tsv', required=True)
    parser.add_argument('-o', '--output-tsv', required=True)
    parser.add_argument('-d', '--align-dir', required=True)
    parser.add_argument('-t', '--index', type=int, required=True)
    parser.add_argument('-r', '--rate', type=float, required=True)
    parser.add_argument('--key', default='phoneme')
    parser.add_argument('--audio-key', default='audio')
    args = parser.parse_args()

    args.align_dir = Path(args.align_dir)

    lines = read_tsv(args.input_tsv)
    
    for line in tqdm(lines):
        audio = line[args.audio_key]
        stem = audio.rsplit('.', maxsplit=1)[0]
        file = args.align_dir/f'{stem}.TextGrid'
        if not file.exists():
            print(f'{file} not exists...')
            continue
        tg = TextGrid.fromFile(file)
        labels = tg2labels(tg[args.index], args.rate)
        line[args.key] = ' '.join(labels)
    
    write_tsv(args.output_tsv, lines)
