# Imports
import torch
import pandas as pd
from mlm import MaskedLanguageModel


# Hyperparameters
DEVICE = "cuda"         # Device - GPU
DIM = 1024               # Model Dimension (Embedding Size)
NHEAD = 8               # Number of Multi-head Attention Heads
NLAYER = 8              # Number of Transformer Encoders in the Stack
DIMFF = 512            # Dimensions of the Transformer Encoder Linear Layers
DIMSUB = 256             # Dimension of Intermediate output layer in FF Network
DROPOUT = 0.            # Model Dropout Value
MASKPROB = 0.01         # Probability of [MASK]-ing a token


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
acid_list = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]",
        "L", "A", "G", "V", "E", "S", "I", "K", "R",
        "D", "T", "P", "N", "Q", "F", "Y", "M", "H", "C",
        "W", "X", "U", "B", "Z", "O"]


# Tokenization of Amino Acid Sequence
def tokenize(X: str, max_len: int = 300) -> list:
    seq_len = len(X)
    padding = (max_len-seq_len) * ["[PAD]"]
    X = [token_dict[x] for x in list(X) + padding]
    return X


# Get data
data = pd.read_csv("./data/proteinas_test.csv")
X = list(data["Sequência"])
X = [x for x in X if len(x) > 200] # Limit data size
X = torch.tensor([tokenize(x) for x in X])
# Y = torch.tensor(data["Hidrofobicidade"])


out = model(X[:1].to(DEVICE))
out.shape
out

xx = torch.argmax(out, dim=2).cpu().numpy()[0]
decoded = [acid_list[x] for x in list(xx)]
decoded