import torch
import torch.nn as nn
import math

"""
Implementation of llama 2 in Pytorch

Notable observations:
    - Llama uses Gated Linear Units in its MLP block. I was wondering why the safetensors had "ffn_gate" tensors, got super confused as this isnt a mixture of experts arch haha
    - Llama uses Silu instead of Relu in the MLP block!
"""


def apply_rotary_embeddings(
    input_tensor: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, position_ids: torch.Tensor
) -> torch.Tensor:
    """
    GPT-NeoX style rotary embeddings. This function splits the input tensor into two halves along the embedding dimension and then rotates them by 90 degrees

    For more details, see:
    - Roformer paper: https://arxiv.org/pdf/2104.09864
    - GPT-NeoX paper: https://arxiv.org/pdf/2204.06745

    This version implements the rotation matrix multiplication directly for each pair of dimensions
    in the input tensor, making it easier to understand but less compute-efficient.

    For each embedding [first_half, second_half], the rotation is applied as follows:
    [ cos(θ)  -sin(θ) ] [ first_half ]
    [ sin(θ)   cos(θ) ] [ second_half ]

    This results in the rotated embedding:
    [ first_half * cos(θ) - second_half * sin(θ) ]
    [ first_half * sin(θ) + second_half * cos(θ) ]

    Args:
        input_tensor (torch.Tensor): Input tensor with shape [batch, seq_len, dim].
        cos (torch.Tensor): Precomputed cos(θ) values for each position, shape [seq_len, dim].
        sin (torch.Tensor): Precomputed sin(θ) values for each position, shape [seq_len, dim].
        position_ids (torch.Tensor): Tensor indicating position indices, shape [seq_len].

    Returns:
        torch.Tensor: The rotated tensor with shape [batch, seq_len, dim].
    """
    cos = cos[position_ids].unsqueeze(1)  # [seq_len, 1, dim]
    sin = sin[position_ids].unsqueeze(1)  # [seq_len, 1, dim]

    # Split the query and key vectors into two halves
    first_half = input_tensor[..., : input_tensor.shape[-1] // 2]  # First half of the input tensor
    second_half = input_tensor[..., input_tensor.shape[-1] // 2 :]  # Second half of the input tensor

    # Apply the rotation to the query vectors
    first_half_rotated = first_half * cos - second_half * sin  # q1 * cos(θ) - q2 * sin(θ)
    second_half_rotated = first_half * sin + second_half * cos  # q1 * sin(θ) + q2 * cos(θ)
    rotated_result = torch.cat([first_half_rotated, second_half_rotated], dim=-1)  # Combine the rotated halves

    return rotated_result


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
        self.cos, self.sin = generate_rotation_magnitudes(
            max_seq_len=max_seq_len,
            embedding_dim=self.head_dim,
            device=self.device,
            dtype=torch.float16,
        )
        self.mask = torch.triu(
            torch.ones((max_seq_len, max_seq_len), device=self.device),
            diagonal=1,
        ).unsqueeze(0)  # [1, query_length, total_seq_length]
        self.w_q = nn.Linear(embedding_dim, num_query_heads * head_dim, device=device, bias=False)
        self.w_k = nn.Linear(embedding_dim, num_kv_heads * head_dim, device=device, bias=False)
        self.w_v = nn.Linear(embedding_dim, num_kv_heads * head_dim, device=device, bias=False)
        self.w_o = nn.Linear(num_query_heads * head_dim, embedding_dim, device=device, bias=False)
        self.kv_cache = None

    def forward(self, x):
        query_length = x.shape[0]  # [query_length, emb_dim] during prefill, [1, emb_dim] during generation
        # Calculate query, key, and value
        query = self.w_q(x)  # [query_length, num_query_heads * head_dim] = [query_length, 32*64] = [query_length, 2048]
        key = self.w_k(x)  # [query_length, num_kv_heads * head_dim] = [1, 4*64] = [query_length, 256]
        value = self.w_v(x)  # Same as key
        # Seperate out heads
        query = query.view(query_length, self.num_query_heads, self.head_dim)  # [query_length, 32, 64]
        key = key.view(query_length, self.num_kv_heads, self.head_dim)  # [query_length, 4, 64]
        value = value.view(query_length, self.num_kv_heads, self.head_dim)

        # KV cache lookup
        if self.kv_cache is not None:
            key = torch.cat([self.kv_cache["key"], key], dim=0)  # [total_seq_length, num_kv_heads, head_dim]
            value = torch.cat([self.kv_cache["value"], value], dim=0)
        # Update KV Cache
        self.kv_cache = {"key": key, "value": value}
        total_seq_length = key.shape[0]

        # Apply rotary embeddings
        position_ids = torch.arange(0, total_seq_length, device=self.device)
        query = apply_rotary_embeddings(query, self.cos, self.sin, position_ids[-query_length:])
        key = apply_rotary_embeddings(key, self.cos, self.sin, position_ids)

        # repeat each key/value tensor (num_query_heads/num_kv_heads) times to match the number of query heads
        # (total_seq_length, num_query_heads, head_dim)
        key = key.repeat_interleave(repeats=self.num_query_heads // self.num_kv_heads, dim=1)
        value = value.repeat_interleave(repeats=self.num_query_heads // self.num_kv_heads, dim=1)

        query = query.transpose(0, 1)  # (num_query_heads, query_length, head_dim)
        key = key.transpose(0, 1)  # (num_query_heads, total_seq_length, head_dim)
        value = value.transpose(0, 1)  # (num_query_heads, total_seq_length, head_dim)

        attention_scores = torch.matmul(query, key.transpose(1, 2)) / math.sqrt(self.head_dim)
        # (num_query_heads, query_length, total_seq_length)

        mask = self.mask[
            :, total_seq_length - query_length : total_seq_length, :total_seq_length
        ]  # [1, query_length, total_seq, length]
        attention_scores = attention_scores.masked_fill(mask == 1, float("-inf"))
        attention_scores = nn.functional.softmax(attention_scores, dim=-1).to(value.dtype)
        out = torch.matmul(attention_scores, value)  # (num_query_heads, query_length, head_dim)

        out = out.transpose(0, 1)  # (query_length, num_query_heads, head_dim)
        out = out.reshape(x.shape[0], self.embedding_dim)  # (query_length, embedding_dim)
        out = self.w_o(out)

        return out  # (query_length, embedding_dim)


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
        max_seq_len,
        intermediate_size,
        device,
    ):
        super(DecoderLayer, self).__init__()
        self.grouped_query_attention = GroupedQueryAttention(
            num_attention_heads, head_dim, num_kv_heads, len_embedding, max_seq_len, device=device
        )
        self.attention_norm = RMSNorm(len_embedding, device=device)
        self.feedforward_norm = RMSNorm(len_embedding, device=device)
        self.feedforward = GatedLinearUnit(len_embedding, intermediate_size, device=device)
        self.device = device

    def forward(self, x):
        # 1: Normalize input
        attention_normalized_x = self.attention_norm.forward(x)
        # 2: MultiHead Self Attention
        self_attention = self.grouped_query_attention.forward(attention_normalized_x)
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
        max_seq_len,
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
                    max_seq_len,
                    intermediate_size,
                    device,
                )
                for _ in range(num_decoder_layers)
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
        return torch.argmax(logits, dim=-1)  # get the token index for the next most likely token in the sequence
