from dataclasses import dataclass
import S2TDataset

@dataclass
class DataItem(S2TDataset.DataItem):

    audio_path: str
    src_text: str
    tgt_text: str

class Dataset(S2TDataset.Dataset):

    Name = "example"

    def __init__(self, *args):
        super().__init__(*args)

        self.data = []
        for i in range(10):
            item = DataItem(
                f'audio_path_{i}',
                f'src_text_{i}',
                f'tgt_text_{i}'
            )
            self.data.append(item)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, id: int) -> DataItem:

        return self.data[id]
