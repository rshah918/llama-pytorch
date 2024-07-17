import torch
import torch.nn as nn
import math

"""
Implementation of llama 2 in Pytorch

Notable observations:
    - Llama 7B does NOT use grouped query attention. GQA is purely an efficiency optimization that slightly hurts performance. 7B is small enough to not need it.
    - Llama uses Gated Linear Units in its MLP block. I was wondering why the safetensors had "ffn_gate" tensors, got super confused as this isnt a mixture of experts arch haha
    - Llama uses Silu instead of Relu in the MLP block
"""


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]  # first half of the embeddings
    x2 = x[..., x.shape[-1] // 2 :]  # second half of the embeddings
    return torch.cat((-x2, x1), dim=-1)  # rotate embeddings


def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def generate_sin_cos(max_seq_len, embedding_dim, device, dtype):
    """
    Calculate vector rotation magnitudes
    Theta calulation from the paper: https://arxiv.org/pdf/2104.09864v5
        Θ = {θi = 10000^−2(i−1)/d , i ∈ [1, 2, ..., d/2]}
        - where i is the position along the sequence dimension, and d is along the embedding dimension
        - theta base is 10000 for Llama architectures
    """
    sequence_positions = (
        torch.arange(max_seq_len, device=device, dtype=dtype).repeat(embedding_dim // 2, 1).float()
    )  # (embedding_dim, max_seq_len)
    theta_base = 10000
    emb_positions = torch.arange(
        start=0, end=embedding_dim, step=2, device=device, dtype=dtype
    ).float()  # (embedding_dim/2)
    theta = 1.0 / (theta_base ** (emb_positions / embedding_dim))  # calculate rotation magnitudes
    theta = sequence_positions * theta.unsqueeze(1)  # unsqueeze for broadcast and complete theta calculation
    theta = theta.transpose(0, 1)  # transpose to get shape (max_seq_len, embedding_dim/2)
    theta = torch.cat((theta, theta), dim=-1)
    cos = torch.cos(theta)[None, None, :, :].to(dtype)  # (1,1,sequence_len, embedding_dim)
    sin = torch.sin(theta)[None, None, :, :].to(dtype)  # (1,1,sequence_len, embedding_dim)
    return cos, sin


class GroupedQueryAttention(nn.Module):
    def __init__(self, num_query_heads, head_dim, num_kv_heads, embedding_dim, len_sequence, device):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.embedding_dim = embedding_dim
        self.head_dim = head_dim
        self.len_sequence = len_sequence
        self.device = device
        self.w_q = nn.Linear(embedding_dim, num_query_heads * head_dim, device=device, bias=False)
        self.w_k = nn.Linear(embedding_dim, num_kv_heads * head_dim, device=device, bias=False)
        self.w_v = nn.Linear(embedding_dim, num_kv_heads * head_dim, device=device, bias=False)
        self.w_o = nn.Linear(num_query_heads * head_dim, embedding_dim, device=device, bias=False)

    def forward(self, x, mask=None):
        # Get query, key, and value
        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)
        # add batch dim
        query = torch.unsqueeze(query, 0)
        key = torch.unsqueeze(key, 0)
        value = torch.unsqueeze(value, 0)
        # (batch, seq, num_heads, head_dim)
        query = query.view(1, x.shape[0], self.num_query_heads, self.head_dim)
        key = key.view(1, x.shape[0], self.num_kv_heads, self.head_dim)
        value = value.view(1, x.shape[0], self.num_kv_heads, self.head_dim)

        cos, sin = generate_sin_cos(
            max_seq_len=self.len_sequence,
            embedding_dim=self.head_dim,
            device=self.device,
            dtype=torch.float16,
        )
        query, key = apply_rotary_pos_emb(query, key, cos, sin, torch.arange(0, x.shape[0]))
        # remove batch dimension
        query = query[0]
        key = key[0]
        value = value[0]
        # repeat each key/value tensor (num_query_heads/num_kv_heads) times to match the number of query heads
        key = torch.repeat_interleave(key, repeats=self.num_query_heads // self.num_kv_heads, dim=1)
        value = torch.repeat_interleave(
            value, repeats=self.num_query_heads // self.num_kv_heads, dim=1
        )  # (seq_len, num_query_heads, head_dim) since num_kv_heads is equal to num_query heads now

        query = query.transpose(0, 1)  # (n_query, seqlen, head_dim)
        key = key.transpose(0, 1)  # (n_query, seqlen, head_dim)
        value = value.transpose(0, 1)  # (n_query, seqlen, head_dim)
        attention_scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(
            self.head_dim
        )  # (num_query_heads, seq_len, seq_len)
        mask = torch.tril(torch.ones((x.shape[0], x.shape[0]), device=query.device)).unsqueeze(
            0
        )  # (1, seq_len, seq_len)
        attention_scores = attention_scores.masked_fill(mask == 0, float("-inf"))
        attention_scores = nn.functional.softmax(attention_scores, -1)
        out = torch.matmul(attention_scores, value)  # (num_query_heads, seq_len, head_dim)

        out = out.transpose(0, 1)  # (seq_len, num_query_heads, head_dim)
        out = out.reshape(x.shape[0], self.embedding_dim)  # (seq_len, embedding_dim)
        out = self.w_o(out)
        return out  # (seq_len, embedding_dim)


class RMSNorm(nn.Module):
    def __init__(self, len_embedding, device):
        super(RMSNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(len_embedding, device=device))
        self.eps = 1e-5

    def forward(self, x):
        # 1: Square every element. Mathematically, this exaggerates large values
        # 2: Calculate the mean, add a small epsilon to prevent divide-by-zero errors
        # 3: Square root
        # 4: Divide each element by the final value from step 3. This normalizes the variance to 1
        # 5: multiply by gamma. NN will update gamma during training
        rms = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        out = self.gamma * rms
        return out


class GatedLinearUnit(nn.Module):
    """
    The Llama 2 7B architecture employs Gated Linear Units (GLUs). The concept involves adding an additional linear layer to selectively downweight certain activations, effectively serving as a "gate".
    This mechanism activates only a subset of the downstream subnetworks, encouraging them to specialize.
    Intuitively, this makes sense. The human brain is made up of specialized regions (vision, audio, reasoning, etc). Gating forces the model to recreate the same architecture.
    """

    def __init__(self, len_embedding, hidden_dimension, device):
        super(GatedLinearUnit, self).__init__()
        self.ffn_gate = nn.Linear(len_embedding, hidden_dimension, device=device, bias=False)
        self.ffn_down_projection = nn.Linear(hidden_dimension, len_embedding, device=device, bias=False)
        self.ffn_up_projection = nn.Linear(len_embedding, hidden_dimension, device=device, bias=False)

    def forward(self, x):
        out = nn.functional.silu(self.ffn_gate(x)) * self.ffn_up_projection(x)  # elementwise multiplication
        out = self.ffn_down_projection(out)
        return out


class DecoderLayer(nn.Module):
    def __init__(
        self,
        num_attention_heads,
        head_dim,
        num_kv_heads,
        len_embedding,
        len_sequence,
        intermediate_size,
        device,
    ):
        super(DecoderLayer, self).__init__()
        self.grouped_query_attention = GroupedQueryAttention(
            num_attention_heads, head_dim, num_kv_heads, len_embedding, len_sequence, device=device
        )
        self.attention_norm = RMSNorm(len_embedding, device=device)
        self.feedforward_norm = RMSNorm(len_embedding, device=device)
        self.feedforward = GatedLinearUnit(len_embedding, intermediate_size, device=device)

    def forward(self, x):
        # generate causal mask
        seq_len = x.shape[0]
        mask = torch.tril(torch.ones((seq_len, seq_len), device=torch.device("mps"))).float()
        mask = mask.masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))
        # 1: Normalize input
        attention_normalized_x = self.attention_norm.forward(x)
        # 2: MultiHead Self Attention
        self_attention = self.grouped_query_attention.forward(attention_normalized_x, mask)
        # 3: Skip connection
        skip_connection = x + self_attention
        # 4: Layer Normalization
        feedforward_norm_output = self.feedforward_norm.forward(skip_connection)
        # 5: Feedforward layer
        feedforward = self.feedforward(feedforward_norm_output)
        # 6: Another skip connection
        out = skip_connection + feedforward
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        num_decoder_layers,
        num_attention_heads,
        num_kv_heads,
        len_embedding,
        len_sequence,
        intermediate_size,
        device,
    ):
        super(Decoder, self).__init__()
        self.device = device
        # Chain together multiple decoder layers.
        self.decoder_layers = nn.Sequential(
            *[
                DecoderLayer(
                    num_attention_heads,
                    len_embedding // num_attention_heads,
                    num_kv_heads,
                    len_embedding,
                    len_sequence,
                    intermediate_size,
                    device,
                )
                for i in range(num_decoder_layers)
            ]
        )
        # Output layer creates a probability distribution across your vocabulary for each token in the input sequence.
        self.output_layer = nn.Linear(len_embedding, vocab_size, bias=False, device=device)
        self.norm = RMSNorm(len_embedding=len_embedding, device=device)
        self.embeddings = nn.Embedding(vocab_size, len_embedding, device=device)

    def forward(self, x):
        x = x.to(self.device)
        x = self.embeddings(x)
        decoder_layers_output = self.decoder_layers(x)
        output_norm = self.norm(decoder_layers_output)
        logits = self.output_layer(output_norm).float()  # shape: [seq_length, vocab_size]
        logits = logits[-1, :]
        probabilities = nn.functional.softmax(logits, dim=-1)  # shape: [seq_length, vocab_size]
        return torch.argmax(probabilities, dim=-1)
