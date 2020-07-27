import torch
import random
import os
import time

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# seeds
def seed():
    seed = 42
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train(model, tokenizer, optimizer, dataset, args):
    model = model.to(device)

    ## Hyperparameters
    batch_size = 4
    gradient_accumulation_steps = 8
    epochs = 3
    learning_rate = 3e-5
    weight_decay = 1e-2
    eps = qe-6
    num_warmup_steps = 0.08
    num_training_steps = int((len(dataset.dataset) / (batch_size * gradient_accumulation_steps)) * epochs)

    no_decay = ["bias"]
    optimizer_grouped_parameters = [
        {'params':[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         "weight_decay": weight_decay}
    ]


class NSP(torch.nn.Module):
    def __init__(self, model):
        super(NSP, self).__init__()
        self.model = model

        # Pool hidden layers corresponding to the first token of the input sequence ([CLS])

        self.tanh = torch.nn.Tanh()
        self.pool = torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)

        self.seq_prediction = torch.nn.Linear(self.model.config.hidden_size, 2)

    def forward(self, input_ids, token_type_ids, attention_mask, cls_pos = 0):
        # the position of CLS being encoded is different for different models
        # cls_pos = -1 : encoded at the end of the sequence (eg. XLNet)
        # cls_pos = 0 : encoded at the beginning of the sequence (eg. BERT)
        output = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        pooled_output = self.pool(self.tanh(output[0][:,cls_pos]))

        return self.seq_prediction(pooled_output)
