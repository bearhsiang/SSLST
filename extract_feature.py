import torchaudio
import torch
import argparse
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import csv

class TSVDataset(Dataset):

    def __init__(self, tsv, path_key, audio_dir, target_sample_rate=16000):

        self.data = []
        self.path_key = path_key
        self.audio_dir = Path(audio_dir)
        self.target_sample_rate = target_sample_rate

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
    parser.add_argument('-p', '--path-key', required=True)
    parser.add_argument('-s', '--src-key', default=None)
    parser.add_argument('-t', '--tgt-key', required=True)
    parser.add_argument('-a', '--audio-dir', required=True)
    parser.add_argument('-d', '--output-dir', required=True)
    parser.add_argument('-g', '--gpu', action='store_true')
    parser.add_argument('-f', '--feature', required=True)
    parser.add_argument('-r', '--sample-rate', default=16000)
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('-n', '--num_workers', default=1, type=int)
    args = parser.parse_args()

    device = 'cuda' if args.gpu else 'cpu'
    feature_list = torch.hub.list('s3prl/s3prl')

    if args.feature not in feature_list:
        print(f'feature "{args.feature}" is not available. Please choose one of {feature_list}')
        exit(1)

    dataset = TSVDataset(
        args.input_tsv,
        args.path_key,
        args.audio_dir,
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

    for batch in tqdm(dataloader):

        batch['wav'] = [wav.to(device) for wav in batch['wav']]

        with torch.no_grad():
            features = extractor(batch['wav'])['last_hidden_state']

        ratio = max([len(wav) for wav in batch['wav']]) / features.size(1)
        features = [feature[:round(len(feature)*ratio)].cpu().numpy() for feature in features]
    
        for feature, item in zip(features, batch['data']):
            
            ID = item[args.path_key].rsplit('.', maxsplit=1)[0]
            np.save(output_dir/f'{ID}.npy', feature)
            item['feature'] = f'{ID}.npy'
            data.append(item)

    header = data[0].keys()

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
