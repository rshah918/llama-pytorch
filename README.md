# llama-pytorch

Let's build Llama 2 from scratch in PyTorch!

## Overview

This project implements the Llama 2 neural net architecture from scratch in PyTorch. It includes a manually written inference pipeline to showcase how LLM's can be created and used without relying on pre-built inference frameworks. My personal laptop is a 16GB M2 Macbook Air, so we will be using a 1.1 Billion parameter variant of Llama 2, trained by the [TinyLlama team](https://github.com/jzhang38/TinyLlama), deployed to Apple Metal Performance Shaders (mps) 

### Highlights:
- **Grouped Query Attention:** An efficient variant of multihead attention, also used in Llama 3.
- **Gated Linear Unit (GLU):** Newer GPT variants use GLUs in the MLP block of each Decoder layer.
- **Manual Loading of SafeTensors:** Directly handling the model weights for better control and understanding.
- **Custom Inference Pipeline:** The entire pre/post-processing pipeline fully spelled out in code, from text to tokens to embeddings, and back to text.

The purpose of this project is to provide a hands-on learning experience for building and deploying a sophisticated language model. By constructing Llama 2 from the ground up, you will gain insight into the architecture and mechanics of state-of-the-art LLM's, as well as practical experience in managing model weights, tokenization, and inference.

## Setup

1. **Download Model and Tokenizer**
   - Download `model.safetensors` and `tokenizer.model` from [Huggingface](https://huggingface.co/TinyLlama/TinyLlama-1.1B-Chat-v1.0/tree/main).
   - Place these files in the `/models` directory.

2. **Create Virtual Environment**
   - Run `virtualenv .venv` to create a virtual environment.

3. **Activate Virtual Environment**
   - Activate your virtual environment with `source .venv/bin/activate`.

4. **Install Dependencies**
   - Install the required dependencies using `poetry install`.

5. **Start Chatting**
   - Load the weights and start the inference pipeline with `poe start`.

## Demo

[Watch the Demo](https://www.youtube.com/watch?v=virODFK7uMU)

![Screenshot](https://github.com/user-attachments/assets/e87b31c2-0b37-4f36-9ca3-013e7788eaa1)

## Notes

- Ensure you have Python and Poetry installed on your machine.
- Follow the steps in the exact order to avoid any setup issues.

Enjoy building and experimenting with Llama 2!
