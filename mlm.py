# References
# 1. https://github.com/A-Alpha-Bio/alphabind
# 2. https://www.science.org/doi/10.1126/sciadv.adr2641
# 3. https://github.com/lucidrains/mlm-pytorch/blob/master/mlm_pytorch/mlm_pytorch.py
# 4. https://github.com/ZhuiyiTechnology/roformer
# 5. https://nn.labml.ai/transformers/rope/index.html
# 6. https://github.com/lucidrains/rotary-embedding-torch/tree/main/rotary_embedding_torch


# Imports
import torch, math, copy
from torch import nn
from typing import Any
from functools import reduce
import torch.nn.functional as F
from torch.nn.modules.container import ModuleList


###
# MASKING FUNCTIONS
###


# Generate Appropriately Sized Boolean Mask
def mask_with_tokens(t: torch.Tensor, token_ids: set) -> torch.Tensor:
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


###
# STANDARD TRANSFORMER MODEL
###

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
                 dim_feedforward: int = 1028,
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
        n_heads: int, 
        n_layers: int,
        dim_feedforward: int = 1024,
        dim_sublayer: int = 128,
        dropout: float = 0.,
        mask_prob = 0.01,
        mask_token_id = 21,
        pad_token_id = 0,
        mask_ignore_token_ids = []):
        super().__init__()


        # Create a PyTorch Encoder Stack with Embeddings and Positional Encoding
        self.transformer = Tx(
            dim=dim, 
            vocab_size=vocab_size, 
            padding_idx=pad_token_id, 
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

        # Generate Output with 2D Softmax
        x = self.outlayer(x) # (Batch, Seq Len, Vocab Size)

        # Return outputs
        return x


###
# ROTARY EMBEDDING MODEL
###

# RoPE
class RotaryPositionalEmbedding(nn.Module):
    """
    This class implements Rotary Positional Embeddings (RoPE)
    proposed in https://arxiv.org/abs/2104.09864.

    Reference implementation (used for correctness verfication)
    can be found here:
    https://github.com/meta-llama/llama/blob/main/llama/model.py#L80

    In this implementation we cache the embeddings for each position upto
    ``max_seq_len`` by computing this during init.

    Args:
        dim (int): Embedding dimension. This is usually set to the dim of each
            head in the attention module computed as ``embed_dim // num_heads``
        max_seq_len (int): Maximum expected sequence length for the
            model, if exceeded the cached freqs will be recomputed
        base (int): The base for the geometric progression used to compute
            the rotation angles
    """
    def __init__(
        self,
        dim: int,
        max_seq_len: int = 300,
        base: int = 10_000,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.base = base
        self.max_seq_len = max_seq_len
        self.rope_init()

    def rope_init(self) -> None:
        theta = 1.0 / (
            self.base
            ** (torch.arange(0, self.dim, 2)[: (self.dim // 2)].float() / self.dim)
        )
        self.register_buffer("theta", theta, persistent=False)
        self.build_rope_cache(self.max_seq_len)

    def build_rope_cache(self, max_seq_len: int = 4096) -> None:
        seq_idx = torch.arange(
            max_seq_len, dtype=self.theta.dtype, device=self.theta.device
        )
        idx_theta = torch.einsum("i, j -> ij", seq_idx, self.theta).float()
        cache = torch.stack([torch.cos(idx_theta), torch.sin(idx_theta)], dim=-1)
        self.register_buffer("cache", cache, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.size(1)
        rope_cache = self.cache[:seq_len]
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
        rope_cache = rope_cache.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
        x_out = torch.stack(
            [
                xshaped[..., 0] * rope_cache[..., 0]
                - xshaped[..., 1] * rope_cache[..., 1],
                xshaped[..., 1] * rope_cache[..., 0]
                + xshaped[..., 0] * rope_cache[..., 1],
            ],
            -1,
        )
        x_out = x_out.flatten(3)
        return x_out.type_as(x) # tensor has shape [b, s, n_h, h_d]

# Multihead Attention Module
class RotaryMultiheadAttention(nn.Module):

    """
    RotaryMultiheadAttention module is an updated MultiheadAttention mechanism with Rotary Postiional Embeddings applied

    Parameters:
        d_model (int): dimension of the embeddings
        nheads (int): Number of multiattention heads
        dropout_prob (float): Chance of node dropout to prevent overfitting
        max_seq_len (int): Rotary Positional Embedding hyperparameter
        activation (Any): PyTorch activation function for q, k, v
    """

    def __init__(self,
                 d_model: int,
                 nheads: int,
                 dropout_prob: float = 0.1,
                 max_seq_len: int = 300,
                 activation: Any = F.gelu):
        super().__init__()
        
        # Standard Layers
        self.d_model = d_model
        self.nheads = nheads
        self.activation = activation
        self.head_dim = d_model // nheads
        self.dropout = nn.Dropout(dropout_prob)
        self.linear_q = nn.Linear(d_model, d_model, bias=True)
        self.linear_k = nn.Linear(d_model, d_model, bias=True)
        self.linear_v = nn.Linear(d_model, d_model, bias=True)
        self.linear_o = nn.Linear(d_model, d_model, bias=True)
        
        # Rotary Positional Embedding Layers
        self.q_rope = RotaryPositionalEmbedding(d_model // nheads, max_seq_len=max_seq_len)
        self.k_rope = RotaryPositionalEmbedding(d_model // nheads, max_seq_len=max_seq_len)


    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        
        # Create Q, K, V matrices
        bsz, seq_len, _ = q.shape
        q, k, v = self.linear_q(q), self.linear_k(k), self.linear_v(v)
        q, k, v = self.activation(q), self.activation(k), self.activation(v)

        # Apply Rotary Embeddings
        q = q.view(bsz, seq_len, self.nheads, self.head_dim)
        k = k.view(bsz, seq_len, self.nheads, self.head_dim)
        q, k = self.q_rope(q), self.k_rope(k)

        # Reformat for attention
        q = q.view(bsz, seq_len, self.nheads*self.head_dim)
        k = k.view(bsz, seq_len, self.nheads*self.head_dim)
        q = q.view(seq_len, bsz * self.nheads, self.head_dim).transpose(0, 1)
        k = k.view(seq_len, bsz * self.nheads, self.head_dim).transpose(0, 1)
        v = v.view(seq_len, bsz * self.nheads, self.head_dim).transpose(0, 1)

        # Apply attention
        qk = torch.bmm(q, k.transpose(-2, -1))
        scores = qk * math.sqrt(1.0 / float(self.d_model))
        attn_output_weights = F.softmax(scores, dim=-1)
        attn_output_weights = self.dropout(attn_output_weights)
        attn_output = torch.bmm(attn_output_weights, v)
        attn_output = (
            attn_output.transpose(0, 1).contiguous().view(seq_len * bsz, self.d_model)
        )
        attn_output = self.linear_o(attn_output)
        attn_output = self.activation(attn_output)
        attn_output = attn_output.view(bsz, seq_len, self.d_model)

        # Return Outputs
        return attn_output


# Transformer Encoder Module
class RotaryTxEncoder(nn.Module):
    """
    Transformer Encoder module that uses Rotary Embeddings in the multihead attention

    Parameters:
    d_model (int): dimension of the embeddings
    nheads (int): Number of multiattention heads
    dim_feedforward (int): Dimension of the Transformer Encoder Feedforward Layer
    dropout_prob (float): Chance of node dropout to prevent overfitting
    max_seq_len (int): Rotary Positional Embedding hyperparameter
    activation (Any): PyTorch activation function for q, k, v

    """

    def __init__(self, 
                 d_model: int,
                 nheads: int,
                 dim_feedforward: int = 1028,
                 dropout_prob: float = 0.1,
                 max_seq_len: int = 300,
                 activation: Any = F.gelu
                ):
        super().__init__()
        
        self.d_model = d_model
        self.attn = RotaryMultiheadAttention(
            d_model=d_model,
            nheads=nheads,
            dropout_prob=dropout_prob,
            max_seq_len = max_seq_len,
            activation = activation
        )
        self.ff1 = nn.Linear(d_model, dim_feedforward)
        self.ff2 = nn.Linear(dim_feedforward, d_model)
        self.norm_self_attn = nn.LayerNorm([d_model])
        self.norm_ff = nn.LayerNorm([d_model])
        self.dropout = nn.Dropout(dropout_prob)
        self.activation = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        # Self-Attention Portion
        z = self.norm_self_attn(x)
        self_attn = self.attn(z, z, z)
        x = x + self.dropout(self_attn)
        
        # Feedforward Portion
        ff = self.norm_ff(x)
        ff = self.activation(self.ff1(ff))
        ff = self.dropout(ff)
        ff = self.activation(self.ff2(ff))
        x = x + self.dropout(ff)

        # Return Encoder Output
        return x

# Transformer Module
class RotaryTx(nn.Module):
    """
    Rotary Transformer module. Uses the typical transformer setup with RotaryMultiheadAttention

    Parameters:
        vocab_size: number of 'words' in the language dictionary
        padding_idx: integer padding token
        d_model: dimension of the expected tensor (usually last dimension)
        n_heads: Number of heads to use in multihead attention of TransformerEncoderLayer
        n_layers: Number of encoder layers in the TransformerEncoder
        dim_feedforward: Dimension of the Transformer Encoder Feedforward Layer
        dropout_prob: Transformer Dropout
        max_seq_len (int): Rotary Positional Embedding hyperparameter
        activation (Any): torch activation function used in various layers
    """

    def __init__(self, 
                 vocab_size: int,
                 padding_idx: int,
                 d_model: int,
                 nheads: int,
                 nlayers: int,
                 dim_feedforward: int = 1028,
                 dropout_prob: float = 0.1,
                 max_seq_len: int = 300,
                 activation: Any = F.gelu
                ):
        super().__init__()
        self.embedder = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            padding_idx=padding_idx,
        )
        encoder_layer = RotaryTxEncoder(
            d_model=d_model,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dropout_prob=dropout_prob,
            max_seq_len=max_seq_len,
            activation=activation
        )
        self.layers = self._get_clones(encoder_layer, nlayers)
        self.norm = nn.LayerNorm([d_model])

    def _get_clones(self, module: Any, N: int) -> ModuleList:
        return ModuleList([copy.deepcopy(module) for _ in range(N)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.embedder(x)
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# Masked Language Model
class RotaryMLM(nn.Module):
    """
    A Masked Language Model that uses Rotary Embeddings in the transformer multihead attention

    Parameters:
        vocab_size (int): number of 'words' in the language dictionary
        d_model (int): dimension of the expected tensor (usually last dimension)
        nheads (int): Number of heads to use in multihead attention
        nlayers (int): Number of encoder layers in the transformer
        dim_feedforward (int): Number of nodes in the transformer encoder feedforward layer
        dropout_prob (float): Probability of dropout in various layers
        max_seq_len (int): Rotary Positional Embedding hyperparameter
        activation (Any): torch activation function used in various layers
        dim_sublayer (int): size of classification hidden layer
        mask_prob (float): Probability that an individual token is [MASK]-ed
        mask_token_id (int): The integer value of the [MASK] token
        pad_token_id (int): The integer value of the [PAD] token
        mask_ignore_token_ids (list): special tokens that should not be masked (i.e. [CLS], [SEP])
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        nheads: int, 
        nlayers: int,
        dim_feedforward: int = 1024,
        dropout_prob: float = 0.,
        max_seq_len: int = 512,
        activation: Any = F.gelu,
        dim_sublayer: int = 256,
        mask_prob: float = 0.01,
        mask_token_id: int = 24,
        pad_token_id: int = 0,
        mask_ignore_token_ids: list = []):
        super().__init__()


        # Create a PyTorch Encoder Stack with Embeddings and Positional Encoding
        self.transformer = RotaryTx(
            vocab_size = vocab_size,
            padding_idx = pad_token_id,
            d_model = d_model,
            nheads = nheads,
            nlayers = nlayers,
            dim_feedforward = dim_feedforward,
            dropout_prob = dropout_prob,
            max_seq_len = max_seq_len,
            activation = activation
        )

        # Setup Output Layers
        self.linear1 = nn.Linear(d_model, dim_sublayer)
        self.linear2 = nn.Linear(dim_sublayer, vocab_size)
        self.outlayer = nn.Softmax(dim=2)
        self.drop = nn.Dropout(dropout_prob)
        self.activation = activation

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
        x = self.activation(x)
        x = self.drop(x)
        x = self.linear2(x) # (Batch, Seq Len, Vocab Size)

        # Generate Output with 2D Softmax
        x = self.outlayer(x) # (Batch, Seq Len, Vocab Size)

        # Return outputs
        return x
