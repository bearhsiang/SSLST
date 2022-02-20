from dataclasses import dataclass
from torch.utils import data
from pathlib import Path

@dataclass
class Item:

    id: str
    audio: str
    speaker: str
    chapter: str
    transcript: str

class Dataset(data.Dataset):

    _name = "librispeech"
    _splits = ['train-clean-100', 'dev-clean', 'test-clean']

    def __init__(self, data_root: Path, split: str, src_lang: str, tgt_lang: str):
        super().__init__()

        assert split in self.get_splits()
    
        self.data = []

        for file in (data_root/split).glob('*/*/*.trans.txt'):
            with open(file, 'r') as f:
                for line in f:
                    prefix, text = line.strip().split(' ', 1)
                    speaker, chapter, id = prefix.split('-')
                    self.data.append(
                        Item(prefix, 
                        f'{split}/{speaker}/{chapter}/{prefix}.flac',
                        speaker, 
                        chapter,
                        text)
                    )
                    
    @classmethod
    def get_name(cls):
        return cls._name

    @classmethod
    def get_splits(cls):
        return cls._splits

    def name(self):
        return self._name

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id: int) -> Item:
        return self.data[id]
