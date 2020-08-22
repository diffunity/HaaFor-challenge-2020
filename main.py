import os
import time
import argparse
import torch
import seaborn as sns
from transformers import AdamW
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from dataloader import dataloader
from model import NSP, ConvNSP, train, test, seed

def main(args):
    
    print("Loading Dataset...")
    train_set, test_set = dataloader(args.tokenizer, args)
        
    print("Dataset Loaded...")
    print(f"Train dataset size: {len(train_set.dataset)}")
    print(f"Test dataset size: {len(test_set.dataset)}")
    
    tot_training_steps = (len(train_set.dataset) / (args.batch_size * args.gradient_accumulation_steps)) * args.epochs
    print("Total training steps: ",round(tot_training_steps,2))
    args.training_steps = int(tot_training_steps) - args.warmup_steps
    print("Warm up steps: ", args.warmup_steps)
    print("Training steps: ",args.training_steps)

    loss_hist, acc_hist = train(args.model, AdamW, train_set, args)
    print("Training Finished...")

    plt.plot(loss_hist)
    plt.savefig(args.destination_folder+"loss_graph.png")
    print(f"Loss curve graphed at {args.destination_folder}loss_graph.png...")
    
    plt.clf()

    plt.plot(acc_hist)
    plt.savefig(args.destination_folder+"acc_graph.png")
    print(f"Loss curve graphed at {args.destination_folder}acc_graph.png...")

    if args.evaluate:
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
    
    parser.add_argument("--epochs",
                        default=5,
                        type=int,
                        help="Number of training epochs")

    parser.add_argument("--batch_size",
                        default=2,
                        type=int,
                        help="Training batch size")
    
    parser.add_argument("--gradient_accumulation_steps",
                        default=16,
                        type=int,
                        help="gradient accumulaiton steps")

    parser.add_argument("--weight_decay",
                        default=1e-2,
                        type=float,
                        help="Weight Decay")

    parser.add_argument("--lr",
                        default=1e-6,
                        type=float,
                        help="Learning Rate")

    parser.add_argument("--pretrainedPATH",
                        default="./model_pretrained/albert/",
                        type=str,
                        help="Where to load the model from?")

    parser.add_argument("--train_new",
                        default=False,
                        type=bool,
                        help="Train new?")

    parser.add_argument("--warmup_steps",
                        default=320,
                        type=int,
                        help="Warmup steps")
    
    parser.add_argument("--cls_pos",
                        default=0,
                        type=int,
                        help="CLS position for NSP. -1 for XLNET 0 for the rest")
    
    parser.add_argument("--conv",
                        default=False,
                        type=bool,
                        help="Convolutional pooling type?")

# data-related
    parser.add_argument("--data_path",
                        default="./data/augmented_data.csv",
                        type=str)
    
    parser.add_argument("--preprocess",
                        default="data_all_sep",
                        type=str,
                        help="how to construct input?")

    parser.add_argument("--train_size",
                        default=0.7,
                        type=float,
                        help="Training Size (ratio)")

    parser.add_argument("--evaluate",
                        default=True,
                        type=bool,
                        help="evaluate?")

# misc
    parser.add_argument("--destination_folder",
                        type=str,
                        help="Destination folder for output")

    args = parser.parse_args()

    if "albert" in args.model:
        from transformers import AlbertModel, AlbertTokenizer
        model = AlbertModel
        args.tokenizer = AlbertTokenizer.from_pretrained(args.model)
        # args.destination_folder = "./result/albert/"
    elif "xlnet" in args.model:
        from transformers import XLNetModel, XLNetTokenizer
        model = XLNetModel
        args.tokenizer = XLNetTokenizer.from_pretrained(args.model)
        # args.destination_folder = "./result/xlnet/"
    else:
        raise NameError("Wrong model input")

    print("Pretrained path directory exists: ", os.path.isdir(args.pretrainedPATH))
    print("Pretrained path: ", args.pretrainedPATH)
    print("Train new? ", args.train_new)
    
    if os.path.isdir(args.pretrainedPATH) and not args.train_new:
        # continue training from the latest checkpoint
        # point to existing directory at args.pretrainedPATH
        _, folders, files = next(iter(os.walk(args.pretrainedPATH)))
        folders = list(filter(lambda x: x[-4].isdigit(), files))
        last_checkpoint = sorted(folders, key=lambda x: int(x.split("_")[-1][0]))[-1]
        args.save_checkpoint = int(last_checkpoint.split("_")[-1][0])
        print(f"Loading model from saved_checkpoint_{args.save_checkpoint}.pt ...")
        # model.load_state_dict(torch.load(args.pretrainedPATH + f"saved_checkpoint_{args.save_checkpoint}.pt"))
        model = torch.load(args.pretrainedPATH + f"saved_checkpoint_{args.save_checkpoint}.pt")
        # model = model.from_pretrained(
        #     args.pretrainedPATH + f"saved_checkpoint_{args.save_checkpoint}"
        # )
        # print(model.parameters())
        args.save_checkpoint += 1

    elif os.path.isdir(args.pretrainedPATH) and args.train_new:
        # Overwrite training on existing path
        # point to existing directory at args.pretrainedPATH
        print(
            "You have designated existing directory but wish to train a new model. \n\
              The newly trained model will overwrite the existing model. \n\
              Stop if you wish to keep the existing model in the directory"
        )

        model = model.from_pretrained(args.model)
        if args.conv:
            model = ConvNSP(model, args)
        else:
            model = NSP(model)
        args.save_checkpoint = 0

    elif not os.path.isdir(args.pretrainedPATH):
        # 새로운 폴더를 만들고, 새로운 training parameters 저장
        # issue new directory at args.pretrainedPATH
        print(f"Creating new directory for {args.pretrainedPATH}")
        os.mkdir(args.pretrainedPATH)
        model = model.from_pretrained(args.model)
        if args.conv:
            model = ConvNSP(model, args)
        else:
            model = NSP(model)
        args.save_checkpoint = 0
    args.model = model

    seed()
    main(args)