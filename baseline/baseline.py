import os
import time
import argparse
import torch
import sys
import sys
sys.path.insert(1, "../")
from dataloader import dataloader
from model import NSP, train, test, seed
from transformers import AdamW
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os
os.chdir("../")
def main(args):
    
    print("Loading Dataset...")
    train_set, test_set = dataloader(args.tokenizer, args)
        
    print("Dataset Loaded...")
    print(f"Train dataset size: {len(train_set.dataset)}")
    print(f"Test dataset size: {len(test_set.dataset)}")

    gt, pred = test(args.model, test_set, args)

    acc = list(map(lambda x: x[0]==x[1], zip(gt, pred)))

    print(f"Final Accuracy: {sum(acc) / len(acc)}")
    print("Classification Report: ")
    print(classification_report(gt, pred))

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

# model-related
    parser.add_argument("--model",
                        default="albert-base-v2",
                        type=str,
                        help="Model Name")

    parser.add_argument("--batch_size",
                        default=2,
                        type=int,
                        help="Training batch size")

# data-related
    parser.add_argument("--data_path",
                        default="./data/augmented_data.csv",
                        type=str)

    parser.add_argument("--train_size",
                        default=0.9,
                        type=float,
                        help="Training Size (ratio)")

    args = parser.parse_args()

    if "xlnet" in args.model:
        from transformers import XLNetModel, XLNetTokenizer
        args.model = XLNetModel.from_pretrained("xlnet-base-cased")
        args.tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
    elif "albert" in args.model:
        from transformers import AlbertModel, AlbertTokenizer
        args.model = AlbertModel.from_pretrained("albert-large-v2")
        args.tokenizer = AlbertTokenizer.from_pretrained("albert-large-v2")
    args.model = NSP(args.model)

    seed()
    main(args)