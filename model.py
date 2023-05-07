import os
import time
import random
import torch
from torch.nn import CrossEntropyLoss
from transformers import get_cosine_schedule_with_warmup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# seeds
def seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class NSP(torch.nn.Module):
    def __init__(self, model):
        super(NSP, self).__init__()
        self.model = model

        # Pool hidden layers corresponding to the first token of the input sequence ([CLS])

        self.tanh = torch.nn.Tanh()
        self.pool = torch.nn.Linear(self.model.config.hidden_size, self.model.config.hidden_size)
        self.seq_prediction = torch.nn.Linear(self.model.config.hidden_size, 2)

    def forward(self, input_ids, token_type_ids, attention_mask, cls_pos=0):
        # the position of CLS being encoded is different for different models
        # cls_pos = -1 : encoded at the end of the sequence (eg. XLNet)
        # cls_pos = 0 : encoded at the beginning of the sequence (eg. BERT)
        output = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask
        )

        pooled_output = self.pool(self.tanh(output[0][:,cls_pos]))

        return self.seq_prediction(self.tanh(pooled_output))

class ConvNSP(torch.nn.Module):
    def __init__(self, model, args):
        super(ConvNSP, self).__init__()
        self.model = model

        # Pool hidden layers corresponding to the first token of the input sequence ([CLS])

        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.LeakyReLU()
        self.cnn = torch.nn.Conv1d(self.model.config.hidden_size, self.model.config.hidden_size, 1, stride=2)
        self.flatten = torch.nn.Flatten(start_dim=1)
        self.linear = torch.nn.Linear(self.model.config.hidden_size * (self.model.config.num_hidden_layers//2), 2)
        

    def forward(self, input_ids, token_type_ids, attention_mask, cls_pos=0):
        # the position of CLS being encoded is different for different models
        # cls_pos = -1 : encoded at the end of the sequence (eg. XLNet)
        # cls_pos = 0 : encoded at the beginning of the sequence (eg. BERT)
        output = self.model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )

        pooled_output = torch.cat(tuple([i[:,0].unsqueeze(1) for i in output[-1][1:]]), dim=1)
        pooled_output = torch.transpose(pooled_output, 1, 2)
        pooled_output = self.cnn(pooled_output)
        pooled_output = self.flatten(pooled_output)
        pooled_output = self.relu(pooled_output)
        return self.linear(pooled_output)

def opt(optimizer, model, args):
        no_decay = ["bias"]
            
        optimizer_grouped_parameters = [
            {'params':[p for n,p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": args.weight_decay}
        ]

        return optimizer(optimizer_grouped_parameters, lr=args.lr)

def train(model, optimizer, dataset, args):

    model = model.to(device)
    optimizer = opt(optimizer, model, args)
    # scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.training_steps)
    
    model.train()

    loss_hist = []
    acc_hist = []
    accuracy = 0

    print("Begin Training...")
    for epoch in range(args.epochs):
        print(f"Epoch {epoch}")
        ctime = time.time()
        for data_e, (inp, tar) in enumerate(dataset):
            tar = tar.to(device)
            input_ids, token_type_ids, attention_mask = inp["input_ids"].to(device), \
                                                        inp["token_type_ids"].to(device), \
                                                        inp["attention_mask"].to(device)
            
            output = model(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask,
                           cls_pos=args.cls_pos)

            loss = CrossEntropyLoss()(output, tar)

            loss = loss / args.gradient_accumulation_steps
            loss.backward()
            
            accuracy += (tar == output.argmax(1)).type(torch.float).mean() / args.gradient_accumulation_steps

            if not data_e % args.gradient_accumulation_steps:
                loss_hist.append(loss.item())
                optimizer.step()
                # scheduler.step()
                optimizer.zero_grad()
                acc_hist.append(accuracy)
                accuracy = 0

            if not data_e % 2000:
                print(f"Batch {data_e} Loss : {loss.item()}")
                print(f"Ground Truth: {tar.tolist()} \t Predicted: {output.argmax(1).tolist()}")
        
        print(
            f"Time taken for epoch{epoch+1} : {round( (time.time() - ctime) / 60, 2 )} MINUTES"
        )
        torch.save(model, args.pretrainedPATH + f"saved_checkpoint_{args.save_checkpoint}.pt")
        # model.save_pretrained(
        #     args.pretrainedPATH + f"saved_checkpoint_{args.save_checkpoint}"
        # )
        print(
            f"Model saved at {args.pretrainedPATH}saved_checkpoint_{args.save_checkpoint}"
        )
        args.save_checkpoint += 1

    return loss_hist, acc_hist


def test(model, dataset, args):
    model = model.to(device)
    model.eval()
    gt = []
    pred = []

    print("Begin Evaluation...")
    ctime = time.time()

    with torch.no_grad():
        for data_e, (inp, tar) in enumerate(dataset):
            
            input_ids, token_type_ids, attention_mask = inp["input_ids"].to(device), \
                                                        inp["token_type_ids"].to(device), \
                                                        inp["attention_mask"].to(device)
            
            output = model(input_ids=input_ids,
                            token_type_ids=token_type_ids,
                            attention_mask=attention_mask)
            
            gt.extend(tar.tolist())
            pred.extend(output.argmax(1).tolist())
            
            if not data_e % 2000:
                print(f"Data batch {data_e}")
                print(f"Ground Truth: {tar.tolist()} \t Predicted: {output.argmax(1).tolist()}")
        print(f"Evaluation time for dataset size {len(dataset.dataset)} : {round( (time.time() - ctime) / 60, 2 )} MINUTES")
    return gt, pred


def evaluate(model, dataset, args):
    model = model.to(device)
    model.eval()
    pred = []
    
    print(f"Begin evaluation on {args.file_path}")
    ctime = time.time()
    
    with torch.no_grad():
        for data_e, inp in enumerate(dataset):

            input_ids, token_type_ids, attention_mask = inp["input_ids"].to(device), \
                                                        inp["token_type_ids"].to(device), \
                                                        inp["attention_mask"].to(device)

            output = model(input_ids=input_ids,
                           token_type_ids=token_type_ids,
                           attention_mask=attention_mask)
            

            output = torch.eye(2)[torch.argmax(output, dim=1)].type(torch.int).tolist()
            pred.extend(output)

            if not data_e % 2000:
                print(f"Time taken for data batch {data_e}: {round( (time.time() - ctime) / 60, 2)} MINUTES")
                
    return pred
