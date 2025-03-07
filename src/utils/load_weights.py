from mlx.core import load as load_safetensors
import mlx.core as mx
import torch
import torch.types
from tqdm import tqdm
from sentencepiece import SentencePieceProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np
from models.llama2 import Llama


def get_model_size_in_gb(model: torch.nn.Module) -> float:
    total_params = sum(p.numel() for p in model.parameters())
    param_size_bytes = 4  # Assuming float32, which is 4 bytes
    total_size_bytes = total_params * param_size_bytes
    total_size_gb = total_size_bytes / (1024**3)  # Convert bytes to GB
    return total_size_gb


def load_layer_weights(
    layer_index: int, weights, model: Llama, dtype: torch.types._dtype, device: torch.types.Device
) -> None:
    # some tensors dont belong to any layer. I just treat i==-1 as a flag to load those tensors in its own thread
    if layer_index == -1:
        model.output_layer.weight = torch.nn.Parameter(
            torch.as_tensor(
                np.array(weights["lm_head.weight"].astype(mx.float16), copy=False),
                dtype=dtype,
                device=device,
            )
        )
        model.norm.gamma = torch.nn.Parameter(
            torch.as_tensor(
                np.array(weights["model.norm.weight"].astype(mx.float16), copy=False),
                dtype=dtype,
                device=device,
            )
        )
        model.embeddings.weight = torch.nn.Parameter(
            torch.as_tensor(
                np.array(weights["model.embed_tokens.weight"].astype(mx.float16), copy=False),
                dtype=dtype,
                device=device,
            )
        )
        return
    decoder_layer = model.decoder_layers[layer_index]
    decoder_layer.grouped_query_attention.w_q.weight = torch.nn.Parameter(
        torch.as_tensor(
            np.array(
                weights["model.layers." + str(layer_index) + ".self_attn.q_proj.weight"].astype(mx.float16),
                copy=False,
            ),
            dtype=dtype,
            device=device,
        )
    )

    decoder_layer.grouped_query_attention.w_v.weight = torch.nn.Parameter(
        torch.as_tensor(
            np.array(
                weights["model.layers." + str(layer_index) + ".self_attn.v_proj.weight"].astype(mx.float16),
                copy=False,
            ),
            dtype=dtype,
            device=device,
        )
    )

    decoder_layer.grouped_query_attention.w_k.weight = torch.nn.Parameter(
        torch.as_tensor(
            np.array(
                weights["model.layers." + str(layer_index) + ".self_attn.k_proj.weight"].astype(mx.float16),
                copy=False,
            ),
            dtype=dtype,
            device=device,
        )
    )

    decoder_layer.grouped_query_attention.w_o.weight = torch.nn.Parameter(
        torch.as_tensor(
            np.array(
                weights["model.layers." + str(layer_index) + ".self_attn.o_proj.weight"].astype(mx.float16),
                copy=False,
            ),
            dtype=dtype,
            device=device,
        )
    )

    decoder_layer.attention_norm.gamma = torch.nn.Parameter(
        torch.as_tensor(
            np.array(
                weights["model.layers." + str(layer_index) + ".input_layernorm.weight"].astype(mx.float16),
                copy=False,
            ),
            dtype=dtype,
            device=device,
        )
    )

    decoder_layer.feedforward_norm.gamma = torch.nn.Parameter(
        torch.as_tensor(
            np.array(
                weights["model.layers." + str(layer_index) + ".post_attention_layernorm.weight"].astype(mx.float16),
                copy=False,
            ),
            dtype=dtype,
            device=device,
        )
    )

    decoder_layer.feedforward.ffn_gate.weight = torch.nn.Parameter(
        torch.as_tensor(
            np.array(
                weights["model.layers." + str(layer_index) + ".mlp.gate_proj.weight"].astype(mx.float16),
                copy=False,
            ),
            dtype=dtype,
            device=device,
        )
    )

    decoder_layer.feedforward.ffn_up_projection.weight = torch.nn.Parameter(
        torch.as_tensor(
            np.array(
                weights["model.layers." + str(layer_index) + ".mlp.up_proj.weight"].astype(mx.float16),
                copy=False,
            ),
            dtype=dtype,
            device=device,
        )
    )

    decoder_layer.feedforward.ffn_down_projection.weight = torch.nn.Parameter(
        torch.as_tensor(
            np.array(
                weights["model.layers." + str(layer_index) + ".mlp.down_proj.weight"].astype(mx.float16),
                copy=False,
            ),
            dtype=dtype,
            device=device,
        )
    )

    return f"Layer {layer_index} weights loaded"


def load_weights(
    safetensor_path: str, model: torch.nn.Module, device: torch.types.Device, dtype: torch.dtype, max_workers: int = 1
) -> None:
    file_path = safetensor_path
    weights = load_safetensors(file_path, return_metadata=False)

    tensor_names = [x for x in weights.keys() if "layers" not in x]
    print(f"Loading tensors: {tensor_names}")

    # TODO: GIL blocks true parallelism because only 1 thread can execute python bytecode at a time. Use multiprocessing instead for parallel loading
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        # some tensors dont belong to any layer. I just treat i==-1 as a flag to load those tensors in its own thread
        for i in range(-1, len(model.decoder_layers)):
            if i == -1:
                futures.append(executor.submit(load_layer_weights, i, weights, model, dtype, device))
            else:
                futures.append(
                    executor.submit(
                        load_layer_weights,
                        i,
                        {k: v for k, v in weights.items() if f"model.layers.{i}" in k},
                        model,
                        dtype,
                        device,
                    )
                )

        # Outer tqdm for overall progress
        with tqdm(
            total=len(futures),
            desc="Loading weight tensors",
            bar_format="{l_bar}{bar} | {n_fmt}/{total_fmt} layers loaded",
            colour="GREEN",
            position=0,  # Pinning at the bottom
            leave=False,
        ) as outer_pbar:
            for future in as_completed(futures):
                result = future.result()
                outer_pbar.update(1)
                outer_pbar.set_postfix_str(result)  # Update tqdm with layer loaded message


def get_device() -> torch.types.Device:
    mps_device = torch.device("cpu")
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not " "built with MPS enabled.")
        else:
            print(
                "MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine."
            )

    else:
        mps_device = torch.device("mps")
    return mps_device


def get_tokenizer(tokenizer_path: str) -> SentencePieceProcessor:
    tokenizer = SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    return tokenizer
