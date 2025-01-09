# Imports
import torch
import pandas as pd
import torch.nn as nn
from mlm import RotaryMLM
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


# Hyperparameters
DATASET = "pdb"
DEVICE = "cuda"         # Device - GPU
BATCH_SIZE = 4         # Data Batch Size During Training
EPOCHS = 100            # Number of Training Epochs
LEARNING_RATE = 9E-6    # Initial Learning Rate
LR_GAMMA = 0.999         # Learning Rate Decay
DIM = 512               # Model Dimension (Embedding Size)
NHEAD = 8               # Number of Multi-head Attention Heads
NLAYER = 8              # Number of Transformer Encoders in the Stack
DIMFF = 1024            # Dimensions of the Transformer Encoder Linear Layers
DIMSUB = 256             # Dimension of Intermediate output layer in FF Network
DROPOUT = 0.1            # Model Dropout Value
MASKPROB = 0.01         # Probability of [MASK]-ing a token
MAXSEQ = 512              # RoPE Positional Encoding Hyperparameter
VOCABSIZE = 25

# Tokenization Dictionary
token_dict = {
    "[PAD]": 0,
    "A": 1, # alanine
    "R": 2, # arginine
    "N": 3, # asparagine
    "D": 4, # aspartic acid
    "C": 5, # cysteine
    "E": 6, # glutamic acid
    "Q": 7, # glutamine
    "G": 8, # glycine
    "H": 9, # histidine
    "I": 10, # isoleucine
    "L": 11, # leucine
    "K": 12, # lysine
    "M": 13, # methionine
    "F": 14, # phenylalanine
    "P": 15, # proline
    "O": 16, # pyrrolysine
    "U": 17, # selenocysteine
    "S": 18, # serine
    "T": 19, # threonine
    "W": 20, # tryptophan
    "Y": 21, # tyrosine
    "V": 22, # valine
    "B": 23, # unknown
    "J": 23, # unknown
    "X": 23, # unknown
    "Z": 23, # unknown
    "[MASK]": 24,
}


# Tokenization of Amino Acid Sequence
def tokenize(X: str, max_len: int = 512) -> list:
    x_list = list(X)
    seq_len = len(X)
    padding = (max_len-seq_len) * ["[PAD]"]
    tokens = [token_dict[x] for x in x_list + padding]
    return tokens


# Get data
data = pd.read_csv(f"./data/{DATASET}.csv")
X = list(data["sequence"])
X = torch.tensor([tokenize(x, MAXSEQ) for x in X])


# Setup Models and Datasets
model = RotaryMLM(
    vocab_size = VOCABSIZE,
    d_model = DIM,
    nheads = NHEAD,
    nlayers = NLAYER,
    dim_feedforward = DIMFF,
    dropout_prob = DROPOUT,
    max_seq_len = MAXSEQ,
    dim_sublayer = DIMSUB,
    mask_prob = MASKPROB,
    mask_token_id = 24,
    pad_token_id = 0
).to(DEVICE)
# model.load_state_dict(torch.load(f"./model/rotary_{DATASET}.h5"))
train_dataset = TensorDataset(torch.Tensor(X).to(DEVICE))
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)

# MLM Loss Function - Negative Log Probability of Selecting The Correct AA
def mlm_loss(y_hat, y):
    vals = torch.where(y==1, y_hat, 1.)
    loss = -torch.sum(torch.log(vals))
    return loss


# Training Loop
model.train()
loss_profile = []
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=LR_GAMMA)
for epoch in range(EPOCHS):
    train_loss = 0
    for xx in train_dataloader:
        optimizer.zero_grad()
        out = model(xx[0])
        y = F.one_hot(xx[0], num_classes=VOCABSIZE)
        loss = mlm_loss(out, y)
        train_loss = train_loss + loss.item()
        loss.backward()
        optimizer.step()
    loss_profile.append(train_loss)
    print(f"Epoch {epoch+1} - LR: {scheduler.get_lr()} | Loss: {train_loss}")
    scheduler.step()
    torch.save(model.state_dict(), f"./model/rotary_{DATASET}.h5")

pd.DataFrame({"loss":loss_profile}).to_csv(f"{DATASET}_rotary_loss.csv")