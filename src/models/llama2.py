import torch
import torch.nn as nn
import math

"""
Implementation of llama 2 in Pytorch

Notable observations:
    - Llama uses Gated Linear Units in its MLP block. I was wondering why the safetensors had "ffn_gate" tensors, got super confused as this isnt a mixture of experts arch haha
    - Llama uses Silu instead of Relu in the MLP block!
"""


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """
    GPT-NeoX style rotary embeddings.

    This function splits the input tensor into two halves along the embedding dimension and then rotates them by 90 degrees

    For more details, see:
    - Roformer paper: https://arxiv.org/pdf/2104.09864
    - GPT-NeoX paper: https://arxiv.org/pdf/2204.06745

    Args:
        x (torch.Tensor): Input tensor with shape (..., 2 * d), where d is the dimension of the rotary embeddings.

    Returns:
        torch.Tensor: Output tensor with the halves rotated, shape (..., 2 * d).
    """
    x1 = x[..., : x.shape[-1] // 2]  # first half of the embeddings
    x2 = x[..., x.shape[-1] // 2 :]  # second half of the embeddings
    return torch.cat((-x2, x1), dim=-1)  # perform rotation


def apply_rotary_embeddings(
    q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary positional embeddings to the query (q) and key (k) tensors with an explicit rotation matrix.

    This version implements the rotation matrix multiplication directly for each pair of dimensions
    in the query and key vectors, making it easier to understand but less compute-efficient.

    For each embedding [q1, q2], the rotation is applied as follows:
    [ cos(θ)  -sin(θ) ] [ q1 ]
    [ sin(θ)   cos(θ) ] [ q2 ]

    This results in the rotated embedding:
    [ q1 * cos(θ) - q2 * sin(θ) ]
    [ q1 * sin(θ) + q2 * cos(θ) ]

    Args:
        q (torch.Tensor): Query tensor with shape [batch, seq_len, dim].
        k (torch.Tensor): Key tensor with shape [batch, seq_len, dim].
        cos (torch.Tensor): Precomputed cos(θ) values for each position, shape [seq_len, dim].
        sin (torch.Tensor): Precomputed sin(θ) values for each position, shape [seq_len, dim].
        position_ids (torch.Tensor): Tensor indicating position indices, shape [seq_len].

    Returns:
        tuple[torch.Tensor, torch.Tensor]: The rotated query and key tensors,
        both with shape [batch, seq_len, dim].
    """
    cos = cos[position_ids].unsqueeze(1)  # [seq_len, 1, dim]
    sin = sin[position_ids].unsqueeze(1)  # [seq_len, 1, dim]

    cos_first_half = cos[..., : cos.shape[-1] // 2]  # First half of cos(θ)
    cos_second_half = cos[..., cos.shape[-1] // 2 :]  # Second half of cos(θ)
    sin_first_half = sin[..., : sin.shape[-1] // 2]  # First half of sin(θ)
    sin_second_half = sin[..., sin.shape[-1] // 2 :]  # Second half of sin(θ)

    # Split the query and key vectors into two halves
    q1 = q[..., : q.shape[-1] // 2]  # First half of the query
    q2 = q[..., q.shape[-1] // 2 :]  # Second half of the query
    k1 = k[..., : k.shape[-1] // 2]  # First half of the key
    k2 = k[..., k.shape[-1] // 2 :]  # Second half of the key

    # Apply the rotation to the query vectors
    q_rot1 = q1 * cos_first_half - q2 * sin_second_half  # q1 * cos(θ) - q2 * sin(θ)
    q_rot2 = q1 * sin_first_half + q2 * cos_second_half  # q1 * sin(θ) + q2 * cos(θ)
    q_embed = torch.cat([q_rot1, q_rot2], dim=-1)  # Combine the rotated halves

    # Apply the rotation to the key vectors
    k_rot1 = k1 * cos_first_half - k2 * sin_second_half  # k1 * cos(θ) - k2 * sin(θ)
    k_rot2 = k1 * sin_first_half + k2 * cos_second_half  # k1 * sin(θ) + k2 * cos(θ)
    k_embed = torch.cat([k_rot1, k_rot2], dim=-1)  # Combine the rotated halves

    return q_embed, k_embed


def generate_rotation_magnitudes(
    max_seq_len: int, embedding_dim: int, device: torch.device, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Calculate vector rotation magnitudes.

    This function generates the cosine and sine components used for rotary positional embeddings.
    The theta calculation follows the method described in the Roformer paper: https://arxiv.org/pdf/2104.09864v5

    Theta calculation formula:
        Θ = {θ_i = 10000^−2(i−1)/d , i ∈ [1, 2, ..., d/2]}
    where:
        - i is the position along the sequence dimension
        - d is the embedding dimension
        - theta base is 10000 for Llama architectures

    Args:
        max_seq_len (int): Maximum sequence length.
        embedding_dim (int): Embedding dimension, should be even.
        device (torch.device): Device to create the tensors on.
        dtype (torch.dtype): Data type of the tensors.

    Returns:
        tuple: Two tensors (cos, sin) of shape (max_seq_len, embedding_dim), representing the cosine and sine components for the rotary embeddings.
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
    # (sequence_len, embedding_dim)
    cos = torch.cos(theta)
    sin = torch.sin(theta)
    return cos, sin


class GroupedQueryAttention(nn.Module):
    """
    Grouped Query Attention module.

    This module implements grouped query attention, an efficient variant of multihead attention where the number of key/value
    heads (`num_kv_heads`) is fewer than the number of query heads (`num_query_heads`). Each query head independently computes attention
    across grouped key/value heads, concatenates the results, and projects them back to the original embedding dimension.

    Args:
        num_query_heads (int): Number of query attention heads.
        head_dim (int): Dimensionality of each attention head.
        num_kv_heads (int): Number of key-value attention heads.
        embedding_dim (int): Dimensionality of the input embeddings.
        max_seq_len (int): Max length of the input sequence.
        device (torch.device): Device on which to create the module.

    Attributes:
        w_q (nn.Linear): Linear layer for query projection.
        w_k (nn.Linear): Linear layer for key projection.
        w_v (nn.Linear): Linear layer for value projection.
        w_o (nn.Linear): Linear layer for output projection.
    """

    def __init__(self, num_query_heads, head_dim, num_kv_heads, embedding_dim, max_seq_len, device):
        super().__init__()
        self.num_query_heads = num_query_heads
        self.num_kv_heads = num_kv_heads
        self.embedding_dim = embedding_dim
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.device = device
        self.w_q = nn.Linear(embedding_dim, num_query_heads * head_dim, device=device, bias=False)
        self.w_k = nn.Linear(embedding_dim, num_kv_heads * head_dim, device=device, bias=False)
        self.w_v = nn.Linear(embedding_dim, num_kv_heads * head_dim, device=device, bias=False)
        self.w_o = nn.Linear(num_query_heads * head_dim, embedding_dim, device=device, bias=False)

    def forward(self, x, mask=None):
        len_sequence = x.shape[0]
        # Get query, key, and value
        query = self.w_q(x)
        key = self.w_k(x)
        value = self.w_v(x)
        # add batch dim
        query = torch.unsqueeze(query, 0)
        key = torch.unsqueeze(key, 0)
        value = torch.unsqueeze(value, 0)
        # (batch, seq, num_heads, head_dim)
        query = query.view(1, len_sequence, self.num_query_heads, self.head_dim)
        key = key.view(1, len_sequence, self.num_kv_heads, self.head_dim)
        value = value.view(1, len_sequence, self.num_kv_heads, self.head_dim)
        # rotary position embeddings
        cos, sin = generate_rotation_magnitudes(
            max_seq_len=self.max_seq_len,
            embedding_dim=self.head_dim,
            device=self.device,
            dtype=torch.float16,
        )
        query, key = apply_rotary_embeddings(query, key, cos, sin, torch.arange(0, len_sequence))
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
        mask = torch.tril(torch.ones((x.shape[0], len_sequence), device=query.device)).unsqueeze(
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
    Gated Linear Unit (GLU) module used in the Llama architecture.

    GLUs enhance neural network architectures by selectively downweighting activations through an additional linear layer acting as a gate.
    This mechanism encourages specialization in downstream subnetworks, mimicking the modular structure observed in the human brain.
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


class Llama(nn.Module):
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
        super(Llama, self).__init__()
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
        logits = logits[-1, :]  # pull the probability distribution for the last token in the sequence
        probabilities = nn.functional.softmax(logits, dim=-1)  # convert to softmax probability distribution
        return torch.argmax(probabilities, dim=-1)  # get the token index for the next most likely token in the sequence
