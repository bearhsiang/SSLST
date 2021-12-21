from dataclasses import dataclass
from torch.utils import data
from pathlib import Path

@dataclass
class DataItem:

    audio_path: str
    src_text: str
    tgt_text: str

class Dataset(data.Dataset):

    Name = "default"

    def __init__(self, data_root: Path, split: str, src_lang: str, tgt_lang: str):
        super().__init__()

        self.data_root = data_root
        self.split = split
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __len__(self):

        return len(self.data)

    def __getitem__(self, id: int) -> DataItem:

        return self.data[id]

    def __repr__(self) -> str:
        return f'{self.Name}-{self.src_lang}-{self.tgt_lang}'
