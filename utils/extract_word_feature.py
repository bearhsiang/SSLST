import argparse
from pathlib import Path
from data_utils import read_tsv
import numpy as np
import io
from textgrid import TextGrid
from tqdm.auto import tqdm
from collections import defaultdict
import json

def get_npy(zip_dir, zip_name):

    path, offset, size = zip_name.split(':')
    offset, size = int(offset), int(size)
    zip_path = zip_dir/path
    with open(zip_path, 'rb') as f:
        f.seek(offset)
        data = f.read(size)
    f = io.BytesIO(data)
    data = np.load(f)
    return data


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-tsv', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--feature-key', required=True)
    parser.add_argument('--id-key', required=True)
    parser.add_argument('--align-dir', required=True)
    parser.add_argument('--feature-dir', required=True)
    parser.add_argument('--max-total-duration', type=int, default=30)
    parser.add_argument('--layer', default='last')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--index', type=int, default=0)
    args = parser.parse_args()

    args.feature_dir = Path(args.feature_dir)
    args.align_dir = Path(args.align_dir)
    lines = read_tsv(args.input_tsv)

    w2feat = defaultdict(list)

    for line in tqdm(lines):
        feature = line[args.feature_key]
        feature = get_npy(args.feature_dir, feature)
        ID = line[args.id_key]
        file = args.align_dir/f'{ID}.TextGrid'
        if not file.exists():
            print(f'{file} not exists...')
            continue
        tg = TextGrid.fromFile(file)
        if tg.maxTime > args.max_total_duration:
            print(f'duration({tg.maxTime}) > max duration({args.max_total_duration}), skip...')

        ratio = feature.shape[0]/tg.maxTime

        for seg in tg[args.index]:
            label = seg.mark
            if label == '':
                continue
            begin = int(seg.minTime*ratio)
            end = int(seg.maxTime*ratio)
            seq_feat = feature[begin:end].mean(axis=0)
            w2feat[label].append(seq_feat)

    for w in w2feat:
        w2feat[w] = np.array(w2feat[w]).mean(axis=0).tolist()

    with open(args.output_json, 'w') as f:
        json.dump(w2feat, f)

