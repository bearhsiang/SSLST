from dataclasses import dataclass
import S2TDataset
from pathlib import Path

@dataclass
class Item:

    audio: str
    src_text: str
    tgt_text: str


class Dataset(S2TDataset.Dataset):

    _name = "example"
    _splits = ['train', 'dev', 'test']

    def __init__(self, data_root: Path, split: str, src_lang: str, tgt_lang: str):
        super().__init__(data_root, split, src_lang, tgt_lang)

        self.data = []
        for i in range(10):
            item = Item(
                f'audio_{i}',
                f'src_text_{i}',
                f'tgt_text_{i}'
            )
            self.data.append(item)

    @classmethod
    def get_name(cls):
        return cls._name

    @classmethod
    def get_splits(cls):
        return cls._splits

    def __len__(self):
        return len(self.data)

    def __getitem__(self, id: int) -> Item:
        return self.data[id]
