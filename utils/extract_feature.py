import torchaudio
import torch
import argparse
from tqdm.auto import tqdm
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
import csv
import os
import zipfile
from data_utils import read_tsv, write_tsv, is_npy_data
from collections import defaultdict
from tempfile import TemporaryDirectory
import json

def create_zip(source_dir, zip_path):

    paths = list(source_dir.glob("*.npy"))
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_STORED) as f:
        for path in tqdm(paths):
            f.write(path, arcname=path.name)

def get_audio_info(zip_path):

    with zipfile.ZipFile(zip_path, 'r') as f:
        info = f.infolist()
    audio_info = {}
    for i in tqdm(info):
        ID = Path(i.filename).stem
        offset, file_size = i.header_offset+30+len(i.filename), i.file_size
        audio_info[ID] = f'{zip_path.name}:{offset}:{file_size}'
        with open(zip_path, 'rb') as f:
            f.seek(offset)
            data = f.read(file_size)
            assert len(data) > 1 and is_npy_data(data)
    return audio_info

def extract_feature(files, extractor, feature_dir, args):

    data = []

    dataset = TSVDataset(
            files,
            args.audio_key,
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
            
            ID = item[args.id_key]
            np.save(feature_dir/f'{ID}.npy', feature)
            item['n_frames'] = feature.shape[0]
            data.append(item)

    return data

class TSVDataset(Dataset):

    def __init__(self, tsvs, audio_key, audio_dir, max_duration, target_sample_rate=16000):

        self.data = []
        self.audio_key = audio_key
        self.audio_dir = Path(audio_dir)
        self.target_sample_rate = target_sample_rate
        self.max_duration = max_duration

        for tsv in tsvs:
            self.data += read_tsv(tsv)

        self.resamplers = {}

    def __getitem__(self, idx):

        wav_file = self.audio_dir / self.data[idx][self.audio_key]
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
            print(f'sample too long, cut to {self.max_duration} sec. ({int(self.max_duration * self.target_sample_rate)}). {self.data[idx]}')
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
    parser.add_argument('-i', '--input-dir', required=True)
    parser.add_argument('-o', '--output-dir', required=True)
    parser.add_argument('--feature-zip', required=True)
    parser.add_argument('--audio-key', default='audio')
    parser.add_argument('--id-key', default='id')
    parser.add_argument('--tmp-dir', default=None)
    parser.add_argument('-a', '--audio-dir', required=True)
    parser.add_argument('-g', '--gpu', action='store_true')
    parser.add_argument('-f', '--feature', required=True)
    parser.add_argument('-r', '--sample-rate', default=16000)
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('-n', '--num_workers', default=1, type=int)
    parser.add_argument('-m', '--max-duration', default=-1, type=float)
    args = parser.parse_args()

    device = 'cuda' if args.gpu else 'cpu'
    feature_list = torch.hub.list('s3prl/s3prl')

    if args.feature not in feature_list:
        print(f'feature "{args.feature}" is not available. Please choose one of {feature_list}')
        exit(1)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    data = defaultdict(list)

    extractor = torch.hub.load('s3prl/s3prl', args.feature).to(device)
    feat_dim = extractor([torch.zeros(10000).to(device)])['last_hidden_state'].size(-1)

    with TemporaryDirectory(dir=args.tmp_dir) as temp_dir:
        temp_dir = Path(temp_dir)
        print('create temporary directory', temp_dir)
        
        for file in input_dir.glob("*.tsv"):
            split_data = extract_feature([file], extractor, temp_dir, args)
            data[file.name] = split_data

        feature_zip = Path(args.feature_zip)
        create_zip(temp_dir, feature_zip)
    
    audio_info = get_audio_info(feature_zip)

    for file in data:
        for item in data[file]:
            ID = item[args.id_key]
            item['audio'] = audio_info[ID]
        write_tsv(output_dir/file, data[file])

    with open(output_dir/'feat_dim.json', 'w') as f:
        json.dump(feat_dim, f)

