import os
import time
import argparse
import torch
from dataloader import testset_dataloader
from model import NSP, evaluate, seed
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

def main(args):
    
    print("Loading Dataset...")
    dataset = testset_dataloader(args.tokenizer, args)
        
    print("Dataset Loaded...")
    print(f"Dataset size: {len(dataset.dataset)}")
    
    pred = evaluate(args.model, dataset, args)

    result = pd.DataFrame(pred)

    result.to_csv("./evaluation_result/evaluation.csv")

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
                        help="Testing batch size")

    parser.add_argument("--pretrainedPATH",
                        default="./model_pretrained/albert/",
                        type=str,
                        help="Where to load the model from?")

# data-related
    parser.add_argument("--file_path",
                        default="./data/testing.csv",
                        type=str)

    args = parser.parse_args()

    if "albert" in args.model:
        from transformers import AlbertModel, AlbertTokenizer
        model = AlbertModel
        args.tokenizer = AlbertTokenizer.from_pretrained(args.model)
        args.destination_folder = "./result/albert/"
    elif "xlnet" in args.model:
        from transformers import XLNetModel, XLNetTokenizer
        model = XLNetModel
        args.tokenizer = XLNetTokenizer.from_pretrained(args.model)
        args.destination_folder = "./result/xlnet/"
    else:
        raise NameError("Wrong model input")

    print("Pretrained path directory exists: ", os.path.isdir(args.pretrainedPATH))
    print("Pretrained path: ", args.pretrainedPATH)
    
    if os.path.isdir(args.pretrainedPATH)
        # load model from pretrainedPATH
        _, folders, files = next(iter(os.walk(args.pretrainedPATH)))
        folders = list(filter(lambda x: x[-4].isdigit(), files))
        last_checkpoint = sorted(folders, key=lambda x: int(x.split("_")[-1][0]))[-1]
        args.save_checkpoint = int(last_checkpoint.split("_")[-1][0])
        print(f"Loading model from saved_checkpoint_{args.save_checkpoint}.pt ...")
        model = torch.load(args.pretrainedPATH + f"saved_checkpoint_{args.save_checkpoint}.pt")

    args.model = model

    seed()
    main(args)