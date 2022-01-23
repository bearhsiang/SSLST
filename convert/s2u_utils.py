import torch
import torchaudio
import torch.nn as nn
from torch.utils import data
import pandas as pd
import csv
from pathlib import Path

SAMPLE_RATE = 16000

class FeatureExtractor():

    def __init__(self, feature: str, device: str, feature_selection='hidden_states'):
        
        self.extractor = torch.hub.load('s3prl/s3prl', feature).to(device)
        self.extractor.eval()
        self.feature_selection = feature_selection

        with torch.no_grad():
            fake_wavs = [torch.rand(SAMPLE_RATE).to(device)]

        fake_features = self.extractor(fake_wavs)[self.feature_selection]
        self.feat_dim = fake_features[0].size(-1)
        self.frame_rate = 100 if fake_features[0].size(1) > 50 else 50
        self.n_layers = len(fake_features)

    def __call__(self, wavs: list):

        with torch.no_grad():
            feature = self.extractor(wavs)[self.feature_selection]
        
        return feature

    def to(self, device):

        self.extractor = self.extractor.to(device)
    


class AudioDataset(data.Dataset):

    tgt_sr = 16000

    def __init__(self, audio_dir, audio_list):

        self.audio_dir = Path(audio_dir)
        self.audio_list = audio_list
        self.resamplers = {}
    
    @classmethod
    def from_tsv(cls, tsv_file, audio_key, audio_dir):

        data = pd.read_csv(
            tsv_file,
            delimiter='\t',
            quoting=csv.QUOTE_NONE,
        )

        audio_list = list(data[audio_key].unique())

        return cls(audio_dir, audio_list)

    def __len__(self):

        return len(self.audio_list)
    
    def __getitem__(self, index):
        
        audio_file = self.audio_dir/self.audio_list[index]
        wav, sr = torchaudio.load(audio_file)
        if sr != self.tgt_sr:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(
                    orig_freq = sr,
                    new_freq = self.tgt_sr,
                )
            wav = self.resamplers[sr](wav)
        
        wav = torch.mean(wav, dim=0) # to mono
        wav = wav.view(-1)

        return {'name': audio_file, 'wav': wav}

    def collate_fn(self, samples):

        return {
            'names': [sample['name'] for sample in samples],
            'wavs': [sample['wav'] for sample in samples]
        }