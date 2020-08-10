import torch
from torch.utils.data import DataLoader, Dataset, random_split
import pandas as pd

class data_all_sep(Dataset):
    """
    data module
    input: 
        [[before_headline, before_body, after_headline, after_body], ... ]
    output: 
        "[CLS] before_headline [SEP] before_body [SEP] after_headline [SEP] after_body"
    """
    def __init__(self, tokenizer, data):
        super(data_all_sep,self).__init__()
        self.data = data[:,:-1]
        self.label = data[:,-1]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        text1 = self.data[ix][0] + " " + self.tokenizer.sep_token + " " + self.data[ix][1] 
        text2 = self.data[ix][2] + " " + self.tokenizer.sep_token + " " + self.data[ix][3]
        
        tokenized = self.tokenizer(text1, text2, 
                                   return_tensors="pt", 
                                   max_length=512,
                                   truncation=True,
                                   padding="max_length")

        return tokenized, self.label[ix]

class data_add_sep(Dataset):
    """
    data module
    input: 
        [[before_headline, before_body, after_headline, after_body], ... ]
    output: 
        "[CLS] before_headline + before_body [SEP] after_headline + after_body"
    """
    def __init__(self, tokenizer, data):
        super(data_add_sep, self).__init__()
        self.data = data[:,:-1]
        # strip string data
        self.label = data[:,-1]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        text1 = self.data[ix][0] + " " + self.data[ix][1]
        text2 = self.data[ix][2] + " " + self.data[ix][3]
        
        tokenized = self.tokenizer(text1, text2, 
                                   return_tensors="pt", 
                                   max_length=512,
                                   truncation=True,
                                   padding="max_length")

        return tokenized, self.label[ix]

class data_body(Dataset):
    """
    data module
    input: 
        [[before_headline, before_body, after_headline, after_body], ... ]
    output: 
        "[CLS] before_body [SEP] after_body"
    """
    def __init__(self, tokenizer, data):
        super(data_body, self).__init__()
        self.data = data[:,:-1]
        # strip string data
        self.label = data[:,-1]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        text1 = self.data[ix][1]
        text2 = self.data[ix][3]
        
        tokenized = self.tokenizer(text1, text2,  
                                   return_tensors="pt", 
                                   max_length=512,
                                   truncation=True,
                                   padding="max_length")

        return tokenized, self.label[ix]


class data_headline(Dataset):
    """
    data module
    input: 
        [[before_headline, before_body, after_headline, after_body], ... ]
    output: 
        "[CLS] before_headline [SEP] after_headline"
    """
    def __init__(self, tokenizer, data):
        super(data_headline, self).__init__()
        self.data = data[:,:-1]
        # strip string data
        self.label = data[:,-1]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, ix):
        text1 = self.data[ix][0]
        text2 = self.data[ix][2]
        
        tokenized = self.tokenizer(text1, text2, 
                                   return_tensors="pt", 
                                   max_length=512,
                                   truncation=True,
                                   padding="max_length")

        return tokenized


def dataloader(tokenizer, args):
    """
    Loading data
    """
    data = pd.read_csv(args.data, index_col=0).values
    dataset = eval(f"{args.preprocess}(data=data, tokenizer={args.tokenzier})")
    n = len(data)
    train, test = random_split(dataset, [int(n*args.train_size), n-int(n*args.train_size)])

    train_loader = DataLoader(dataset=train,
                              batch_size=args.batch_size, 
                              shuffle=True, 
                              num_workers=4)
    
    test_loader = DataLoader(dataset=test,
                             batch_size=args.batch_size, 
                             shuffle=True, 
                             num_workers=4)
    
    return train_loader, test_loader