from dataclasses import dataclass
import S2TDataset
from pathlib import Path

@dataclass
class Item:

    audio: str
    src_text: str
    tgt_text: str


class Dataset(S2TDataset.Dataset):

    _name = "libritrans"
    _splits = ['train', 'dev', 'test']

    def __init__(self, data_root: Path, split: str, src_lang: str, tgt_lang: str):
        super().__init__(data_root, split, src_lang, tgt_lang)

        meta_file = data_root / split / "alignments.meta"
        src_file = data_root / split / f'{split}.{src_lang}'
        tgt_file = data_root / split / f'{split}.{tgt_lang}'

        with open(meta_file, 'r') as f:
            f.readline()
            audio_files = [f'{line.split()[4]}.wav' for line in f]
        
        with open(src_file, 'r') as f:
            src_texts = [line.strip() for line in f]

        with open(tgt_file, 'r') as f:
            tgt_texts = [line.strip() for line in f]
        
        assert len(audio_files) == len(src_texts)
        assert len(src_texts) == len(tgt_texts)

        self.data = [
            Item(audio, src, tgt) 
            for audio, src, tgt in zip(audio_files, src_texts, tgt_texts)
        ]

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
