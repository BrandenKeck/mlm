# References
# 1. https://github.com/A-Alpha-Bio/alphabind
# 2. https://www.science.org/doi/10.1126/sciadv.adr2641
# 3. https://github.com/lucidrains/mlm-pytorch/blob/master/mlm_pytorch/mlm_pytorch.py
# 4. https://github.com/ZhuiyiTechnology/roformer


# Imports
import torch, math
from torch import nn
from functools import reduce
import torch.nn.functional as F


# Generate Appropriately Sized Boolean Mask
def mask_with_tokens(t: list, token_ids: list) -> torch.Tensor:
    init_no_mask = torch.full_like(t, False, dtype=torch.bool)
    mask = reduce(lambda acc, el: acc | (t == el), token_ids, init_no_mask)
    return mask


# Replace Random Tokens with [MASK]
def get_mask_subset_with_prob(mask: torch.Tensor, prob: float) -> torch.Tensor:
    batch, seq_len, device = *mask.shape, mask.device
    max_masked = math.ceil(prob * seq_len)
    num_tokens = mask.sum(dim=-1, keepdim=True)
    mask_excess = (mask.cumsum(dim=-1) > (num_tokens * prob).ceil())
    mask_excess = mask_excess[:, :max_masked]
    rand = torch.rand((batch, seq_len), device=device).masked_fill(~mask, -1e9)
    _, sampled_indices = rand.topk(max_masked, dim=-1)
    sampled_indices = (sampled_indices + 1).masked_fill_(mask_excess, 0)
    new_mask = torch.zeros((batch, seq_len + 1), device=device)
    new_mask.scatter_(-1, sampled_indices, 1)
    return new_mask[:, 1:].bool()


# Vanilla Positional Encoding
class PositionalEncoding(nn.Module):
    """
    PositionalEncoding module as described in 'Attention is all you need': https://arxiv.org/pdf/1706.03762.pdf

    Parameters:
        dim (int): dimension (usually last dimension) size of the tensor to be passed into the module
        max_len (int): Maximum length of the input (usually dim_1, but dim_0 if not using batch_first)
    """

    def __init__(self, dim: int, max_len: int = 5000):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))

        pe = torch.zeros(1, max_len, dim)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)

        # register so that it is a member but not updated by gradient descent:
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return x


# RoPE
class RoPE(nn.Module):

    def __init__(self):
        pass # TODO


# Transformer Module
class Tx(nn.Module):
    """
    Transformer module.

    Parameters:
        dim: dimension of the expected tensor (usually last dimension)
        vocab_size: number of 'words' in the language dictionary
        padding_idx: integer padding token
        n_heads: Number of heads to use in multihead attention of TransformerEncoderLayer
        n_layers: Number of encoder layers in the TransformerEncoder
        dim_feedforward: Dimension of the Transformer Encoder Feedforward Layer
        dropout: Transformer Dropout
    """

    def __init__(self, 
                 dim: int,
                 vocab_size: int,
                 padding_idx: int,
                 n_heads: int,
                 n_layers: int,
                 dim_feedforward: int = 2048,
                 dropout: float = 0.1
                ):
        super().__init__()
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            padding_idx=padding_idx,
        )
        self.pos_encoder = PositionalEncoding(dim=dim)
        tx_encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim, batch_first=True, nhead=n_heads, activation="gelu", 
            dim_feedforward=dim_feedforward, dropout=dropout
        )
        self.tx_encoder = nn.TransformerEncoder(tx_encoder_layer, num_layers=n_layers)

    def forward(
        self,
        x: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.embedder(x)
        x = self.pos_encoder(x)
        x = self.tx_encoder(x, src_key_padding_mask=padding_mask)
        return x


# Masked Language Model
class MaskedLanguageModel(nn.Module):
    """
    A Masked Language Model

    Parameters:
        dim: dimension of the expected tensor (usually last dimension)
        vocab_size: number of 'words' in the language dictionary
        padding_idx: integer padding token
        n_heads: Number of heads to use in multihead attention of TransformerEncoderLayer
        n_layers: Number of encoder layers in the TransformerEncoder
        mask_prob: Probability that an individual token is [MASK]-ed
        mask_token_id: The integer value of the [MASK] token
        pad_token_id: The integer value of the [PAD] token
        mask_ignore_token_ids: special tokens that should not be masked (i.e. [CLS], [SEP])
    """

    def __init__(
        self,
        dim: int,
        vocab_size: int,
        padding_idx: int,
        n_heads: int, 
        n_layers: int,
        dim_feedforward: int = 1024,
        dim_sublayer: int = 128,
        dropout: float = 0.1,
        mask_prob = 0.25,
        mask_token_id = 4,
        pad_token_id = 0,
        mask_ignore_token_ids = [1, 2, 3]):
        super().__init__()


        # Create a PyTorch Encoder Stack with Embeddings and Positional Encoding
        self.transformer = Tx(
            dim=dim, 
            vocab_size=vocab_size, 
            padding_idx=padding_idx, 
            n_heads=n_heads, 
            n_layers=n_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout)

        # Setup Output Layers
        self.linear1 = nn.Linear(dim, dim_sublayer)
        self.linear2 = nn.Linear(dim_sublayer, vocab_size)
        self.outlayer = nn.Softmax(dim=2)
        self.drop = nn.Dropout(dropout)
        self.gelu = nn.GELU()

        # Store hyperparameters at the class level
        self.mask_prob = mask_prob
        self.pad_token_id = pad_token_id
        self.mask_token_id = mask_token_id
        self.mask_ignore_token_ids = set([*mask_ignore_token_ids, pad_token_id])



    def forward(self, seq: torch.Tensor, **kwargs) -> torch.Tensor:

        # Generates a boolean mask tensor, ignoring special tokens
        no_mask = mask_with_tokens(seq, self.mask_ignore_token_ids)
        mask = get_mask_subset_with_prob(~no_mask, self.mask_prob)

        # [MASK] the input sequence
        masked_seq = seq.clone().detach()
        masked_seq = masked_seq.masked_fill(mask, self.mask_token_id)

        # Pass Masked Data to Transformer to get Embeddings
        x = self.transformer(masked_seq, **kwargs) # (Batch, Seq Len, Embed)

        # Generate Correct Dimensions via Linear Layer
        x = self.linear1(x) # (Batch, Seq Len, Sublayer Size)
        x = self.gelu(x)
        x = self.drop(x)
        x = self.linear2(x) # (Batch, Seq Len, Vocab Size)
        x = self.gelu(x)

        # Generate Output with 2D Softmax
        x = self.outlayer(x) # (Batch, Seq Len, Vocab Size)

        # Return outputs
        return x
