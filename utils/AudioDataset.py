from torch.utils.data import Dataset
import torchaudio
from .data_utils import read_tsv
from pathlib import Path
import torch

class AudioDataset(Dataset):

    def __init__(self, audio_list: list, data_list: list, tgt_sr: int = 16000):

        assert len(audio_list) == len(data_list)

        self.audio_list = audio_list
        self.data_list = data_list
        self.tgt_sr = tgt_sr

        self.resamplers = {}

    def __len__(self):

        return len(self.data_list)

    def __getitem__(self, index):

        wav, sr = torchaudio.load(self.audio_list[index])
        if sr != self.tgt_sr:
            if sr not in self.resamplers:
                self.resamplers[sr] = torchaudio.transforms.Resample(
                    orig_freq = sr,
                    new_freq = self.tgt_sr,
                )
            wav = self.resamplers[sr](wav)
        
        wav = torch.mean(wav, dim=0) # to mono
        wav = wav.view(-1)
        
        return {'wav': wav, 'data': self.data_list[index]}

    def collate_fn(self, samples):

        return {
            'wav': [sample['wav'] for sample in samples],
            'data': [sample['data'] for sample in samples]
        }
    
    @classmethod
    def from_tsv(cls, tsv, audio_key, audio_dir = '', tgt_sr=16000):

        audio_dir = Path(audio_dir)

        data_list = read_tsv(tsv)
        audio_list = [audio_dir/item[audio_key] for item in data_list]

        return cls(audio_list, data_list, tgt_sr)

if __name__ == '__main__':

    dataset = AudioDataset.from_tsv(
        'data/covost_en_de/test.tsv', 
        'audio', 
        audio_dir='/ssd/covost/cv-corpus-6.1-2020-12-11/en/clips/')

    print(dataset[0])