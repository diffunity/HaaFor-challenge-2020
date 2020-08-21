# HaaFor Challenge 2020
**Task**: Given two news articles with headlines, classify if the first article chronologically precedes the second article.

## Data

### **<ins>Data Description</ins>**

### **<ins>Data Preprocessing</ins>**
- Augmentation (same as BERT NSP task)
    - 50% correct sequence (true)
    - 50% incorrect sequence (false)

- Headline and body preprocessing
    1. Concatenate with [SEP] token (best performance)
    2. Train separately
    3. Extract verbs and numbers only

## Model

### Baseline

1. BertModel
    * Accuracy: 0.50

2. BertForNextSentencePrediction
    * Accuracy: 0.51

### Custom headers on models

1. NSP header (as implemented in huggingface.co)
    * newly trained pooling layer for pooler_output

![NSP header structure](imgs/NSP.png)


2. Convolutional header 
    * Concatenate CLS token embeddings at each hidden layer
    * Apply CNN and fully connected layer

![Convolutional NSP header structure](imgs/ConvNSP.png)

### Model Implementation
- Hyperparameters

| Model       | Pretrained   | Batch Size  | Epochs | Learning Rate | Weight decay | Scheduling | Test Accuracy |
| ----------- |------------- | ----------- | ------ | ------------- | ------------ | ---------- | -------- |
| XLNet       | xlnet-base-cased | 64     | 5 | 1e-5 | 1e-2 | None | -------- |
| XLNet       | xlnet-base-cased | 128    | 5 | 1e-5 | 1e-2 | cosine | -------- |
| Albert      | albert-base-v2   | 32     | 5 | 1e-5 | 1e-2 | None | 0.73 |
| Albert      | albert-base-v2   | 128    | 5 | 1e-5 | 1e-2 | cosine | 0.70 |
| Albert      | albert-base-v2   | 64     | 5 | 1e-5 | 1e-2 | None | -------- |
