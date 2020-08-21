import time
import argparse
import torch
import sys
sys.path.insert(1, "../")
from dataloader import dataloader
from model import NSP, train, seed
from transformers import AdamW
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import os
os.chdir("../")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

def test(model, dataset, args):
    # model = NSP(model).to(device)
    model = model.to(device)
    model.eval()

    gt = []
    pred = []

    print("Begin Evaluation...")
    ctime = time.time()
    for data_e, (inp, tar) in enumerate(dataset):
        
        input_ids, token_type_ids, attention_mask = inp["input_ids"].to(device), \
                                                    inp["token_type_ids"].to(device), \
                                                    inp["attention_mask"].to(device)
        
        output = model(input_ids=input_ids,
                        token_type_ids=token_type_ids,
                        attention_mask=attention_mask)[0]
        
        gt.extend(tar.tolist())
        pred.extend(output.argmax(1).tolist())
        
        if not data_e % 2000:
            print(f"Data batch {data_e}")
            print(f"Ground Truth: {tar.tolist()} \t Predicted: {output.argmax(1).tolist()}")
    print(f"Evaluation time for dataset size {len(dataset.dataset)} : {round( (time.time() - ctime) / 60, 2 )} MINUTES")
    return gt, pred

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
                        default=0.5,
                        type=float,
                        help="Training Size (ratio)")

    args = parser.parse_args()

    from transformers import BertForNextSentencePrediction, BertTokenizer

    args.model = BertForNextSentencePrediction.from_pretrained("bert-base-uncased")
    args.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    seed()
    main(args)