from mlx.core import load as load_safetensors
import torch
from llama2 import Decoder
from time import time
from sentencepiece import SentencePieceProcessor
import torch._dynamo
torch._dynamo.config.suppress_errors = True


def get_model_size_in_gb(model):
    total_params = sum(p.numel() for p in model.parameters())
    param_size_bytes = 4  # Assuming float32, which is 4 bytes
    total_size_bytes = total_params * param_size_bytes
    total_size_gb = total_size_bytes / (1024 ** 3)  # Convert bytes to GB
    return total_size_gb


def load_weights(safetensor_path: str, model: torch.nn.Module, device, dtype: torch.dtype):
    file_path = safetensor_path
    weights = load_safetensors(file_path, return_metadata=False)
    print([x for x in weights.keys() if "layers" not in x])
    for i in range(len(model.decoder_layers)):
        decoder_layer = model.decoder_layers[i]
        decoder_layer.grouped_query_attention.w_q.weight = torch.nn.Parameter( torch.as_tensor(weights["model.layers."+str(i)+".self_attn.q_proj.weight"].tolist(), dtype=dtype, device=device))
        decoder_layer.grouped_query_attention.w_v.weight = torch.nn.Parameter( torch.as_tensor(weights["model.layers."+str(i)+".self_attn.v_proj.weight"].tolist(), dtype=dtype, device=device))
        decoder_layer.grouped_query_attention.w_k.weight = torch.nn.Parameter( torch.as_tensor(weights["model.layers."+str(i)+".self_attn.k_proj.weight"].tolist(), dtype=dtype, device=device))
        decoder_layer.grouped_query_attention.w_o.weight = torch.nn.Parameter( torch.as_tensor(weights["model.layers."+str(i)+".self_attn.o_proj.weight"].tolist(), dtype=dtype, device=device))
        decoder_layer.attention_norm.gamma = torch.nn.Parameter( torch.as_tensor(weights["model.layers."+str(i)+".input_layernorm.weight"].tolist(), dtype=dtype, device=device))
        decoder_layer.feedforward_norm.gamma = torch.nn.Parameter( torch.as_tensor(weights["model.layers."+str(i)+".post_attention_layernorm.weight"].tolist(), dtype=dtype, device=device))
        decoder_layer.feedforward.ffn_gate.weight = torch.nn.Parameter( torch.as_tensor(weights["model.layers."+str(i)+".mlp.gate_proj.weight"].tolist(), dtype=dtype, device=device))
        decoder_layer.feedforward.ffn_up_projection.weight = torch.nn.Parameter( torch.as_tensor(weights["model.layers."+str(i)+".mlp.up_proj.weight"].tolist(), dtype=dtype, device=device))
        decoder_layer.feedforward.ffn_down_projection.weight = torch.nn.Parameter( torch.as_tensor(weights["model.layers."+str(i)+".mlp.down_proj.weight"].tolist(), dtype=dtype, device=device))
    model.output_layer.weight = torch.nn.Parameter( torch.as_tensor(weights["lm_head.weight"].tolist(), dtype=dtype, device=device))
    model.norm.gamma = torch.nn.Parameter( torch.as_tensor(weights["model.norm.weight"].tolist(), dtype=dtype, device=device))
    model.embeddings.weight = torch.nn.Parameter( torch.as_tensor(weights["model.embed_tokens.weight"].tolist(), dtype=dtype, device=device))


def get_device():
    mps_device=torch.device("cpu")
    if not torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                "built with MPS enabled.")
        else:
            print("MPS not available because the current MacOS version is not 12.3+ "
                "and/or you do not have an MPS-enabled device on this machine.")

    else:
        mps_device = torch.device("mps")
    return mps_device


def get_tokenizer(tokenizer_path: str):
    tokenizer = SentencePieceProcessor()
    tokenizer.load(tokenizer_path)
    return tokenizer


with torch.inference_mode():
    model = Decoder(vocab_size=32000, num_decoder_layers=22, num_attention_heads=32, num_kv_heads=4, len_embedding=2048, len_sequence=2048, intermediate_size=5632, device=get_device())
    model = torch.compile(model=model, backend="aot_eager")
    load_weights(safetensor_path="../models/model.safetensors", model=model, device=get_device(), dtype=torch.float16)
    tokenizer = get_tokenizer(tokenizer_path="../models/tokenizer.model")

    while(True):
        prompt = input("Ask me anything: ")
        formatted_prompt = (
        f"""
<|user|>
{prompt}</s>
<|assistant|>"""
    )
        tokenized_prompt: list[int] = tokenizer.encode(formatted_prompt)
        model_response = []
        stop_strings = ["<|user|>", "<|system|>", "</s>"]
        for i in range(2048-len(tokenized_prompt)):
            tik = time()
            out = model.forward(torch.as_tensor(tokenized_prompt+model_response, device=get_device()))
            tok = time()
            elapsed = tok-tik
            out = out.item()
            model_response.append(out)
            decoded_model_response = tokenizer.decode(model_response)
            if any(stop_string in decoded_model_response for stop_string in stop_strings):
                break
        print(decoded_model_response)
