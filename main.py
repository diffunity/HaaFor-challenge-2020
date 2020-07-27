import configparser
import os
import tensorflow as tf
import time
# from transformers import (
#     AdamW, get_linear_schedule_with_warmup,
#     BertForNextSentenceClassification, BertTokenizer,
#     AlbertModel, AlbertTokenizer,
#     DistilBertModel, DistilBertTokenizer,
#     ElectraModel, ElectraTokenizer,
#     XLNetModel, XLNetTokenizer,
#     RobertaModel, RobertaTokenizer
# )
if __name__ == "__main__":

    config = configparser.ConfigParser()

    config.read("./config.ini")
