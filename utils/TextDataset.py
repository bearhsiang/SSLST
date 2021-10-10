from torch.utils.data import Dataset
from .data_utils import read_tsv

def intersection(lst1, lst2):
    return list(set(lst1) & set(lst2))

class TextDataset(Dataset):

    @classmethod
    def from_tsv(cls, tsv_file, key, word_list = None):
        data = read_tsv(tsv_file)
        text_list = []
        for item in data:
            text = item[key]
            if word_list == None or len(intersection(text.strip().split(), word_list)) > 0:
                text_list.append(text)
        # text_list = [item[key] for item in data]
        return cls(text_list)

    def __init__(self, text_list):
        self.data = text_list
    
    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)