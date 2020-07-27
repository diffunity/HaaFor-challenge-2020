import torch
from torch.utils.data import DataLoader, Dataset
import argparse

class DATA(Dataset):
    def __init__(self, tokenizer, data):
        super(DATA,self).__init__()
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):

def dataloader(tokenizer, args):
