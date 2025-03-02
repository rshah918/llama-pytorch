from time import time
from utils.load_weights import get_device, get_tokenizer, load_weights
import torch
from models.llama2 import Llama
import gradio as gr
import torch._dynamo
from typing import Generator

torch._dynamo.config.suppress_errors = True


def format_prompt(prompt: str, history: list[dict[str, str]]) -> str:
    formatted_message_history = ""
    for message in history:
        if message["role"] == "user":
            formatted_message_history += f"""<|user|>{message["content"]}</s>"""
        else:
            formatted_message_history += f"""<|assistant|>{message["content"]}</s>"""
    if prompt != "":
        formatted_message_history += f"<|user|>{prompt}</s>"

    formatted_prompt = f"""{formatted_message_history}<|assistant|>"""
    return formatted_prompt


@torch.inference_mode()
def chat(prompt: str, history: list[dict[str, str]]) -> Generator[str, None, None]:
    tokenizer = get_tokenizer(tokenizer_path="src/models/tokenizer.model")
    formatted_prompt = format_prompt(prompt, history)
    tokenized_prompt: list[int] = tokenizer.encode(formatted_prompt)
    model_response = []
    stop_strings = ["<|user|>", "<|system|>", "</s>"]
    for _ in range(2048 - len(tokenized_prompt)):
        tik = time()
        out = model.forward(torch.as_tensor(tokenized_prompt + model_response, device=get_device()))
        tok = time()
        elapsed = tok - tik
        print((1 / elapsed), " Tokens per second")
        out = out.item()
        model_response.append(out)
        decoded_model_response = tokenizer.decode(model_response)
        if any(stop_string in decoded_model_response for stop_string in stop_strings):
            break
        yield decoded_model_response


# address a gradio bug where the chat window doesnt fill up the screen, css taken from: https://github.com/gradio-app/gradio/issues/4001#issuecomment-1636785196
CSS = """
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""

model = Llama(
    vocab_size=32000,
    num_decoder_layers=22,
    num_attention_heads=32,
    num_kv_heads=4,
    len_embedding=2048,
    len_sequence=2048,
    intermediate_size=5632,
    device=get_device(),
)
model = torch.compile(model=model, backend="aot_eager")
tik = time()
load_weights(
    safetensor_path="src/models/model.safetensors",
    model=model,
    device=get_device(),
    dtype=torch.float16,
)
tok = time()
print("Loaded weights in: ", tok - tik, " seconds")
gr.ChatInterface(
    chat,
    type="messages",
    css=CSS,
    title="Llama 1.1B",
    description="Chat with a 1.1 Billion parameter variant of Llama 2!",
    fill_height=True,
).launch()
