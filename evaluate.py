# Imports
import torch
import pandas as pd
from mlm import RotaryMLM


# Hyperparameters
DATASET = "pdb"
DEVICE = "cpu"         # Device - GPU
DIM = 512               # Model Dimension (Embedding Size)
NHEAD = 8               # Number of Multi-head Attention Heads
NLAYER = 8              # Number of Transformer Encoders in the Stack
DIMFF = 1024            # Dimensions of the Transformer Encoder Linear Layers
DIMSUB = 256             # Dimension of Intermediate output layer in FF Network
DROPOUT = 0.1            # Model Dropout Value
MASKPROB = 0.005         # Probability of [MASK]-ing a token
MAXSEQ = 512              # RoPE Positional Encoding Hyperparameter
VOCABSIZE = 25


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
    mask_token_id = 21,
    pad_token_id = 0
).to(DEVICE)
model.load_state_dict(torch.load(f"./model/rotary_{DATASET}.h5"))
model.eval()

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
acid_list = [
    "[PAD]",
    "A","R","N","D",
    "C","E","Q","G",
    "H","I","L","K","M","F",
    "P","O","U","S","T","W","Y",
    "V","[UNK]","[MASK]"
]


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


# out = model(X[:1].to(DEVICE))
# out.shape
# print(out)

# xx = torch.argmax(out, dim=2).cpu().numpy()[0]
# decoded = [acid_list[x] for x in list(xx)]
# print(decoded)

print(token_dict["R"])
xx = X[:1]
xx[0][4] = 24
out = model(xx.to(DEVICE))
print(out[0, 4, :])

import numpy as np
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
dist = out[0, 4, :].detach().cpu().numpy()
ax.bar([acid_list[ii] for ii in range(25)], list(dist))
ax.set_xticklabels([acid_list[ii] for ii in range(25)], rotation=90)
plt.savefig("dist_test.png")
plt.close()

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
pdb_125 = list(pd.read_csv("pdb_loss_1.2e-5.csv")["loss"])
pdbr_125 = list(pd.read_csv("pdb_rotary_loss_1.2e-5.csv")["loss"])
pdb_84 = list(pd.read_csv("pdb_loss_8e-4.csv")["loss"])
pdbr_84 = list(pd.read_csv("pdb_rotary_loss_8e-4.csv")["loss"])
loss_data = pd.DataFrame({
    "Epoch": range(2, len(pdb_125)),
    "T1 Loss": pdb_125[2:],
    "T2 Loss": pdb_84[2:],
    "R1 Loss": pdbr_125[2:],
    "R2 Loss": pdbr_84[2:]
})
loss_data.plot(x="Epoch", y=["T1 Loss", "T2 Loss", "R1 Loss", "R2 Loss"],
                xlabel="Epoch",
                ylabel="Loss"
                )
plt.savefig("losses.png")
plt.close()