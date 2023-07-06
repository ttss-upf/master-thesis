import torch
import pandas as pd
from torch.utils.data import Dataset
from torchtext.data.functional import sentencepiece_numericalizer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CNNDataset(Dataset):
    def __init__(self, path, sp_model):
        self.path = path
        self.df = pd.read_csv(path)
        self.sp_model = sp_model
        self.sp_id_generator = sentencepiece_numericalizer(self.sp_model)
        self.bos_id = 1
        self.eos_id = 2
        self.pad_id = 3
        self.tsp_id = 4

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, ind):
        text = self.df.iloc[ind]['text']
        query = self.df.iloc[ind]['query']
        summary = self.df.iloc[ind]['summary']
        # print(query)
        ids = list(self.sp_id_generator([text, query, summary]))

        ids[0].insert(0, self.bos_id)
        ids[0].append(self.eos_id)
        ids[1].insert(0, self.bos_id)
        ids[1].append(self.eos_id)
        ids[2].insert(0, self.bos_id)
        ids[2].append(self.eos_id)
        text = ids[0]
        query = ids[1]
        # print(query)
        # print('='*10)
        tgt = ids[2]

        text = torch.tensor(list(text), dtype=torch.long).to(device)
        query = torch.tensor(list(query), dtype=torch.long).to(device)
        tgt = torch.tensor(list(tgt), dtype=torch.long).to(device)

        return text, query, tgt