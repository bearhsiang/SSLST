from dataclasses import dataclass
import S2TDataset
from pathlib import Path

import pandas as pd
import csv

@dataclass
class Item:

    path: str
    sentence: str
    translation: str
    client_id: str

class Dataset(S2TDataset.Dataset):

    _name = "covost2"
    _splits = ['train', 'dev', 'test']

    def __init__(self, data_root: Path, split: str, src_lang: str, tgt_lang: str):
        super().__init__(data_root, split, src_lang, tgt_lang)

        tsv_file = data_root / f'covost_v2.{src_lang}_{tgt_lang}.{split}.tsv'
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        self.raw_data = pd.read_csv(tsv_file, 
            delimiter='\t',
            quoting=csv.QUOTE_NONE,
        )
        # drop samples contain REMOVE
        self.raw_data = self.raw_data[~self.raw_data['translation'].astype(str).str.contains('REMOVE')]

        self.raw_data['path'] = f'{src_lang}/clips/' + self.raw_data['path']

    def name(self):
        return f'{self._name}-{self.src_lang}-{self.tgt_lang}'

    @classmethod
    def get_name(cls):
        return cls._name

    @classmethod
    def get_splits(cls):
        return cls._splits

    def __len__(self):

        return len(self.raw_data)

    def __getitem__(self, id: int) -> Item:

        return Item(**self.raw_data.iloc[id]) 
    

