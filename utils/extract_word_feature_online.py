import argparse
from pathlib import Path
from data_utils import read_tsv
import numpy as np
import io
import torch
from textgrid import TextGrid
from tqdm.auto import tqdm
from collections import defaultdict
import json
from AudioDataset import AudioDataset
from FeatureExtractor import FeatureExtractor
from torch.utils.data import DataLoader

def get_tg_from_batch(batch, args):
    
    tgs = []

    for i in range(len(batch['data'])):
        audio = batch['data'][i][args.audio_key]
        stem = audio.split('.')[0]
        tg_file = args.align_dir / f'{stem}.TextGrid'
        tg = TextGrid.fromFile(tg_file)
        tgs.append(tg)

    return tgs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-tsv', required=True)
    parser.add_argument('--audio-key', required=True)
    parser.add_argument('--audio-dir', default='')
    parser.add_argument('--align-dir', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--feature', required=True)
    parser.add_argument('--max-total-duration', type=int, default=30)
    parser.add_argument('--layer', default='last')
    parser.add_argument('--gpu', action='store_true')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()

    args.align_dir = Path(args.align_dir)

    device = 'cuda' if args.gpu else 'cpu'

    dataset = AudioDataset.from_tsv(args.input_tsv, args.audio_key, args.audio_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
    )
    extractor = FeatureExtractor(args.feature, device)

    w2feat = {}

    for batch in tqdm(dataloader):

        wavs = batch['wav']
        data = batch['data']
        tgs = get_tg_from_batch(batch, args)
        wavs = [wav.to(device) for wav in wavs]

        with torch.no_grad():
            features = extractor(wavs)['hidden_states']
        
        assert len(features) == extractor.n_layers
        tqdm.write(f'len of features {len(features)}')

        for i in range(len(data)):

            tg = tgs[i]
            feature = [layer_feat[i] for layer_feat in features]

            for seg in tg[args.index]:
                label = seg.mark
                if label == '':
                    continue
                begin = int(seg.minTime*extractor.frame_rate)
                end = int(seg.maxTime*extractor.frame_rate)

                if label not in w2feat:
                    w2feat[label] = [[] for _ in range(extractor.n_layers)]

                for layer in range(extractor.n_layers):
                    w_feat = feature[layer][begin:end].mean(dim=0)
                    w2feat[label][layer].append(w_feat)

    for w in w2feat:
        for layer in range(extractor.n_layers):
            w2feat[w][layer] = torch.stack(w2feat[w][layer]).mean(dim=0).tolist()
        
    with open(args.output_json, 'w') as f:
        json.dump(w2feat, f)








    # w2feat = defaultdict(list)

    # for line in tqdm(lines):
    #     feature = line[args.feature_key]
    #     feature = get_npy(args.feature_dir, feature)
    #     ID = line[args.id_key]
    #     file = args.align_dir/f'{ID}.TextGrid'
    #     if not file.exists():
    #         print(f'{file} not exists...')
    #         continue
    #     tg = TextGrid.fromFile(file)
    #     if tg.maxTime > args.max_total_duration:
    #         print(f'duration({tg.maxTime}) > max duration({args.max_total_duration}), skip...')

    #     ratio = feature.shape[0]/tg.maxTime

    #     for seg in tg[args.index]:
    #         label = seg.mark
    #         if label == '':
    #             continue
    #         begin = int(seg.minTime*ratio)
    #         end = int(seg.maxTime*ratio)
    #         seq_feat = feature[begin:end].mean(axis=0)
    #         w2feat[label].append(seq_feat)

    # for w in w2feat:
    #     w2feat[w] = np.array(w2feat[w]).mean(axis=0).tolist()

    # with open(args.output_json, 'w') as f:
    #     json.dump(w2feat, f)

