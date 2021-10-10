import torchaudio
import torch
import argparse
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import csv
import os

class TSVDataset(Dataset):

    def __init__(self, tsv, path_key, audio_dir, max_duration, target_sample_rate=16000):

        self.data = []
        self.path_key = path_key
        self.audio_dir = Path(audio_dir)
        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration

        with open(tsv, 'r') as f:
            reader = csv.DictReader(
                f,
                delimiter='\t',
                quotechar=None,
                doublequote=False,
                lineterminator='\n',
                quoting=csv.QUOTE_NONE,
            )
            for line in reader:
                self.data.append(line)

        self.resamplers = {}

    def __getitem__(self, idx):

        wav_file = self.audio_dir / self.data[idx][self.path_key]
        wav, sr = torchaudio.load(wav_file)
        
        if sr != self.target_sample_rate:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(
                    orig_freq = sr,
                    new_freq = self.target_sample_rate,
                )
            wav = self.resamplers[sr](wav)
        
        wav = torch.mean(wav, dim=0) # to mono
        wav = wav.view(-1)
        
        if self.max_duration > 0 and wav.size(0) > self.max_duration * self.target_sample_rate:
            print(f'sample too long, cut to {self.max_duration} sec. ({int(self.max_duration * self.target_sample_rate)}. {self.data[idx]})')
            wav = wav[:int(self.max_duration*self.target_sample_rate)]

        return {
            'wav': wav,
            'data': self.data[idx]
        }

    def __len__(self):
        return len(self.data)

    def collate_fn(self, samples):

        return {
            'wav': [sample['wav'] for sample in samples],
            'data': [sample['data'] for sample in samples],
        }


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-tsv', required=True)
    parser.add_argument('-o', '--output-tsv', required=True)
    parser.add_argument('-F', '--feat_file', default=None)
    parser.add_argument('-p', '--path-key', required=True)
    parser.add_argument('-a', '--audio-dir', required=True)
    parser.add_argument('-d', '--output-dir', required=True)
    parser.add_argument('-g', '--gpu', action='store_true')
    parser.add_argument('-f', '--feature', required=True)
    parser.add_argument('-r', '--sample-rate', default=16000)
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('-n', '--num_workers', default=1, type=int)
    parser.add_argument('-m', '--max-duration', default=-1, type=float)
    args = parser.parse_args()

    if os.path.isfile(args.output_tsv):
        print(f'{args.output_tsv} is exist, skip...')
        exit(1)

    device = 'cuda' if args.gpu else 'cpu'
    feature_list = torch.hub.list('s3prl/s3prl')

    if args.feature not in feature_list:
        print(f'feature "{args.feature}" is not available. Please choose one of {feature_list}')
        exit(1)

    dataset = TSVDataset(
        args.input_tsv,
        args.path_key,
        args.audio_dir,
        args.max_duration,
        args.sample_rate,
    )

    dataloader = DataLoader(
        dataset, 
        batch_size = args.batch_size,
        num_workers = args.num_workers,
        collate_fn = dataset.collate_fn,
    )

    extractor = torch.hub.load('s3prl/s3prl', args.feature).to(device)

    data = []
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    feat = -1

    for batch in tqdm(dataloader):

        batch['wav'] = [wav.to(device) for wav in batch['wav']] # List[ (wav_len,) ]
        wav_lengths = [wav.size(0) for wav in batch['wav']]

        with torch.no_grad():
            try: 
                features = extractor(batch['wav'])['last_hidden_state'] # B x L x C
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    print(wav_lengths)
                raise e

        ratio =  features.size(1) / max(wav_lengths)
        features = [feature[:round(wav_len*ratio)].cpu().numpy() for wav_len, feature in zip(wav_lengths, features)]

        for feature, item in zip(features, batch['data']):
            
            ID = item[args.path_key].rsplit('.', maxsplit=1)[0]
            np.save(output_dir/f'{ID}.npy', feature)
            item['feature'] = f'{ID}.npy'
            item['n_frames'] = feature.shape[0]
            if feat == -1:
                feat = feature.shape[1]
            assert feat == feature.shape[1]
            data.append(item)

    header = data[0].keys()

    if args.feat_file != None:
        with open(args.feat_file, 'w') as f:
            print(feat, file=f)

    with open(args.output_tsv, 'w') as f:
        writer = csv.DictWriter(
            f,
            delimiter='\t',
            quotechar=None,
            doublequote=False,
            lineterminator='\n',
            quoting=csv.QUOTE_NONE,
            fieldnames=header,
        )
        writer.writeheader()
        writer.writerows(data)
