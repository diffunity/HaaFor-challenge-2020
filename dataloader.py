import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset, random_split

class data_all_sep(Dataset):
    """
    Dataset module
    input: 
        [[before_headline, before_body, after_headline, after_body, label], ... ]
    output: 
        "[CLS] before_headline [SEP] before_body [SEP] after_headline [SEP] after_body", label
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

        tokenized["input_ids"] = tokenized["input_ids"].squeeze(0)
        tokenized["token_type_ids"] = tokenized["token_type_ids"].squeeze(0)
        tokenized["attention_mask"] = tokenized["attention_mask"].squeeze(0)
        return tokenized, self.label[ix]

class test_data_all_sep(Dataset):
    """
    Dataset module for test set 
    input: 
        [[before_headline, before_body, after_headline, after_body], ... ]
    output: 
        "[CLS] before_headline [SEP] before_body [SEP] after_headline [SEP] after_body"
    """
    def __init__(self, tokenizer, data):
        super(test_data_all_sep,self).__init__()
        self.data = data
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

        tokenized["input_ids"] = tokenized["input_ids"].squeeze(0)
        tokenized["token_type_ids"] = tokenized["token_type_ids"].squeeze(0)
        tokenized["attention_mask"] = tokenized["attention_mask"].squeeze(0)
        return tokenized


def dataloader(tokenizer, args):
    """
    Loading data
    """
    data = pd.read_csv(args.data_path, index_col=0).fillna("NONE").values
    # dataset = eval(f"{args.preprocess}(data=data, tokenizer={tokenizer})")
    dataset = data_all_sep(tokenizer=tokenizer, data=data)
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

def testset_dataloader(tokenizer, args):
    data = pd.read_csv(args.file_path, index_col=0).fillna("NONE").values
    dataset = test_data_all_sep(tokenizer=tokenizer, data=data)
    test_loader = DataLoader(dataset=dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=4)
    return test_loader


if __name__ == "__main__":
    # tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")

    class args:
        data_path = "./data/augmented_data.csv"
        train_size = 0.5
        batch_size = 5

    args = args()

    train_set, test_set = dataloader(tokenizer, args)
    inp, tar = next(iter(train_set))
    
    input_ids = inp["input_ids"].squeeze()


    print(input_ids)
    for inpp in input_ids:
        print(tokenizer.decode(inpp))
    print(tar)
