import argparse
import pandas as pd

# def main(args):

#     data = pd.read_csv(args.original_fname, index_col=0)

#     aug_data = data.copy()
#     tmp = aug_data.iloc[-1,:][["BEFORE_HEADLINE","BEFORE_BODY"]].values
#     aug_data.iloc[1:,:][["AFTER_HEADLINE","AFTER_BODY"]] = aug_data.iloc[:-1,:][["AFTER_HEADLINE","AFTER_BODY"]].values
#     aug_data.iloc[0,:][["AFTER_HEADLINE","AFTER_BODY"]] = tmp
    
#     data["Label"] = 1
#     aug_data["Label"] = 0
    
#     pd.concat([data, aug_data]).reset_index(drop=True).to_csv(args.augmented_fname)

def main(args):
    data = pd.read_csv(args.original_fname, index_col=0)
    
    aug_data = data.copy()
    aug_data.columns = list(data)[2:] + list(data)[:2]
    data["Label"] = 1
    aug_data["Label"] = 0
    data.append(aug_data).reset_index(drop=True).to_csv(args.augmented_fname)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()

    parser.add_argument("--original_fname",
                        default="./data/training.csv",
                        type=str,
                        help="File name of the original data")

    parser.add_argument("--augmented_fname",
                        default="./data/augmented_data.csv",
                        type=str,
                        help="File name for the augmented data")

    args = parser.parse_args()

    print(f"Loading data from {args.original_fname} ...")
    main(args)
    print(f"Augmented data saved at {args.augmented_fname} ...")
