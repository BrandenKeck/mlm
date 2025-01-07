# Imports
import torch
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from mlm import MaskedLanguageModel
from torch.utils.data import TensorDataset, DataLoader


# Hyperparameters
DEVICE = "cuda"         # Device - GPU
BATCH_SIZE = 4         # Data Batch Size During Training
EPOCHS = 100            # Number of Training Epochs
LEARNING_RATE = 1E-5    # Initial Learning Rate
LR_GAMMA = 0.999        # Learning Rate Decay
DIM = 1024               # Model Dimension (Embedding Size)
NHEAD = 8               # Number of Multi-head Attention Heads
NLAYER = 8              # Number of Transformer Encoders in the Stack
DIMFF = 512            # Dimensions of the Transformer Encoder Linear Layers
DIMSUB = 256             # Dimension of Intermediate output layer in FF Network
DROPOUT = 0.            # Model Dropout Value
MASKPROB = 0.01         # Probability of [MASK]-ing a token


# ProtBERT Standard Tokenization Dictionary
token_dict = {
    "[PAD]": 0, "[UNK]": 1, "[CLS]": 2,
    "[SEP]": 3, "[MASK]": 4,
    "A":6, "B":27, "C":23, "D":14, "E":9, "F":19,
    "G":7, "H":22, "I":11, "J":1, "K":12, "L":5,
    "M":21, "N":17, "O":29, "P":16, "Q":18, "R":13,
    "S":10, "T":15, "U":26, "V":8, "W":24, "X":25,
    "Y":20, "Z":28
}


# Tokenization of Amino Acid Sequence
def tokenize(X: str, max_len: int = 300) -> list:
    seq_len = len(X)
    padding = (max_len-seq_len) * ["[PAD]"]
    X = [token_dict[x] for x in list(X) + padding]
    return X


# Get data
data = pd.read_csv("./data/proteinas_train.csv")
X = list(data["Sequência"])
X = [x for x in X if len(x) > 200] # Limit data size
X = torch.tensor([tokenize(x) for x in X])
# Y = torch.tensor(data["Hidrofobicidade"])


# Setup Models and Datasets
model = MaskedLanguageModel(
    dim=DIM, 
    vocab_size=29, 
    padding_idx=0, 
    n_heads=NHEAD, 
    n_layers=NLAYER,
    dim_feedforward=DIMFF,
    dim_sublayer=DIMSUB,
    dropout=DROPOUT,
    mask_prob=MASKPROB,
    mask_token_id=4,
    pad_token_id=0,
    mask_ignore_token_ids = [1, 2, 3]
).to(DEVICE)
model.load_state_dict(torch.load(f"./model/proteinas.h5"))
train_dataset = TensorDataset(torch.Tensor(X).to(DEVICE))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)


# Training Loop
model.train()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)
for epoch in range(EPOCHS):
    train_loss = 0
    for xx in train_dataloader:
        optimizer.zero_grad()
        out = model(xx[0])
        out = out.view(-1, 29, 300)
        loss = criterion(out, xx[0])
        train_loss = train_loss + loss.item()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1} - LR: {scheduler.get_lr()} | Loss: {train_loss}")
    scheduler.step()

# Save the Model
torch.save(model.state_dict(), f"./model/proteinas.h5")