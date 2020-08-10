import argparse
import pandas as pd

def main(args):

    data = pd.read_csv(args.original_fname, index_col=0)

    aug_data = data.copy()
    tmp = aug_data.iloc[-1,:][["HEADLINE_2","BODY_2"]].values
    aug_data.iloc[1:,:][["HEADLINE_2","BODY_2"]] = aug_data.iloc[:-1,:][["HEADLINE_2","BODY_2"]].values
    aug_data.iloc[0,:][["HEADLINE_2","BODY_2"]] = tmp
    
    data["Label"] = 1
    aug_data["Label"] = 0
    
    pd.concat([data, aug_data]).reset_index(drop=True).to_csv(args.augmented_fname)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--original_fname",
                        default="./train.csv",
                        type=str,
                        help="File name of the original data")

    parser.add_argument("--augmented_fname",
                        default="./augmented_data.csv",
                        type=str,
                        help="File name for the augmented data")

    args = parser.parse_args()

    print(f"Loading data from {args.original_fname} ...")
    main(args)
    print(f"Augmented data saved at {args.augmented_fname} ...")
