from time import time
from src.load_weights import get_device, get_tokenizer, load_weights
import torch
from llama2 import Decoder
import gradio as gr
import torch._dynamo
torch._dynamo.config.suppress_errors = True



def chat(prompt, history):
    tokenizer = get_tokenizer(tokenizer_path="../models/tokenizer.model")
    formatted_prompt = (
        f"""<|user|>{prompt}</s><|assistant|>"""
    )
    tokenized_prompt: list[int] = tokenizer.encode(formatted_prompt)
    model_response = []
    stop_strings = ["<|user|>", "<|system|>", "</s>"]
    for i in range(2048-len(tokenized_prompt)):
        tik = time()
        with torch.inference_mode():
            out = model.forward(torch.as_tensor(tokenized_prompt+model_response, device=get_device()))
        tok = time()
        elapsed = tok-tik
        print((1/elapsed), " Tokens per second")
        out = out.item()
        model_response.append(out)
        decoded_model_response = tokenizer.decode(model_response)
        if any(stop_string in decoded_model_response for stop_string in stop_strings):
            break
        yield decoded_model_response

#address a gradio bug where the chat window doesnt fill up the screen, css taken from: https://github.com/gradio-app/gradio/issues/4001#issuecomment-1636785196
CSS = """
.contain { display: flex; flex-direction: column; }
.gradio-container { height: 100vh !important; }
#component-0 { height: 100%; }
#chatbot { flex-grow: 1; overflow: auto;}
"""

with torch.inference_mode():
    model = Decoder(vocab_size=32000, num_decoder_layers=22, num_attention_heads=32, num_kv_heads=4, len_embedding=2048, len_sequence=2048, intermediate_size=5632, device=get_device())
    model = torch.compile(model=model, backend="aot_eager")
    tik = time()
    load_weights(safetensor_path="../models/model.safetensors", model=model, device=get_device(), dtype=torch.float16)
    tok = time()
    print("Loaded weights in: ", tok-tik, " seconds")
    gr.ChatInterface(chat, css=CSS, title="Llama 1.1B", description="Chat with a 1.1 Billion parameter variant of Llama 2!", fill_height = True).launch()
