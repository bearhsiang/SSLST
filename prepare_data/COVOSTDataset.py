from dataclasses import dataclass
import S2TDataset

import pandas

@dataclass
class DataItem(S2TDataset.DataItem):

    client_id: str

class Dataset(S2TDataset.Dataset):

    Name = "covost"

    def __init__(self, *args):
        super().__init__(*args)

        tsv_file = self.data_root / f'covost_v2.{self.src_lang}_{self.tgt_lang}.tsv'
        raw_data = pandas.read_csv(tsv_file, 
            delimiter='\t',
        )

        print(raw_data)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, id: int) -> DataItem:

        return self.data[id]
    

