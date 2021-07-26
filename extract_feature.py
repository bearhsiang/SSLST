import torchaudio
import torch
import argparse
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
import csv

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-tsv', required=True)
    parser.add_argument('-o', '--output-tsv', required=True)
    parser.add_argument('-p', '--path-key', required=True)
    parser.add_argument('-s', '--src-key', default=None)
    parser.add_argument('-t', '--tgt-key', required=True)
    parser.add_argument('-a', '--audio-dir', required=True)
    parser.add_argument('-d', '--output-dir', required=True)
    parser.add_argument('-g', '--gpu', action='store_true')
    parser.add_argument('-f', '--feature', required=True)
    parser.add_argument('-r', '--sample-rate', default=16000)
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    args = parser.parse_args()

    device = 'cuda' if args.gpu else 'cpu'
    feature_list = torch.hub.list('s3prl/s3prl')
    if args.feature not in feature_list:
        print(f'feature "{args.feature}" is not available. Please choose one of {feature_list}')
        exit(1)

    audio_dir = Path(args.audio_dir)
    output_dir = Path(args.output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    extractor = torch.hub.load('s3prl/s3prl', args.feature).to(device)

    lines = []
    with open(args.input_tsv, 'r') as f:
        reader = csv.DictReader(
            f,
            delimiter='\t',
            quotechar=None,
            doublequote=False,
            lineterminator='\n',
            quoting=csv.QUOTE_NONE,
        )
        for line in reader:
            lines.append(line)


    resamplers = {}

    data = []

    for offset in tqdm(range(0, len(lines), args.batch_size)):

        batch = lines[offset: offset+args.batch_size]
        src_batch = []

        for line in batch:
            src, sr = torchaudio.load(audio_dir/line[args.path_key])
            if sr not in resamplers:
                resamplers[sr] = torchaudio.transforms.Resample(
                    orig_freq = sr,
                    new_freq = args.sample_rate,
                )
            src = resamplers[sr](src)
            src_batch.append(src)

        with torch.no_grad():
            feature_batch = extractor(src_batch)['last_hidden_state']

        for line, feature in zip(batch, feature_batch):

            index = line[args.path_key].rsplit('.', maxsplit=1)[0]
            np.save(output_dir/index, feature)

            item = {
                'id': index,
                'path': f'{index}.npy',
                'n_frames': feature.size(0),
                'tgt_text': line[args.tgt_key]
            }

            if args.src_key != None:
                item['src_text'] = line[args.src_key]

            data.append(item)

    header = ['id', 'path', 'n_frames', 'tgt_text']
    if args.src_key != None:
        header.append('src_text')

    with open(args.output_tsv, 'w') as f:
        writer = csv.DictWriter(
            f,
            delimiter='\t',
            quotechar=None,
            doublequote=False,
            lineterminator='\n',
            quoting=csv.QUOTE_NONE,
            fieldnames=header
        )
        writer.writeheader()
        writer.writerows(data)

