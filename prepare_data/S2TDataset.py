from dataclasses import dataclass
from torch.utils import data
from pathlib import Path

@dataclass
class DataItem:

    audio: str
    src_text: str
    tgt_text: str

class Dataset(data.Dataset):

    def __init__(self, data_root: Path, split: str, src_lang: str, tgt_lang: str):
        super().__init__()

        assert split in self.get_splits()

    @classmethod
    def get_name(cls):
        raise NotImplementedError

    @classmethod
    def get_splits(cls):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, id: int) -> DataItem:
        raise NotImplementedError
