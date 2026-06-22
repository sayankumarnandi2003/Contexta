import streamlit as st
import torch
import os
from gpt import GPTLanguageModel, decode, encode, device 
from data_loader import clean_text

st.set_page_config(page_title="Local Custom GPT", layout="centered")
st.title("🇮🇳 Custom Local GPT Generator")
st.write("This model generates text word-by-word based on your custom history/geography text file.")

@st.cache_resource
def load_model():
    model = GPTLanguageModel()
    if os.path.exists('model.pth'):
        model.load_state_dict(torch.load('model.pth', map_location=device))
    else:
        st.warning("model.pth not found. Generating with untrained weights. Please run `python train.py` first.")
    model.eval()
    return model

model = load_model()

prompt_text = st.text_input("Enter a starting phrase/seed text:", "The physical features of India ")
tokens_to_generate = st.slider("Number of words to generate", 10, 500, 100)
temperature = st.slider("Temperature (lower = more deterministic, higher = more random/creative)", 0.1, 2.0, 0.8)
top_k = st.slider("Top-K Sampling (0 = disable)", 0, 100, 50)
repetition_penalty = st.slider("Repetition Penalty (1.0 = none, >1.0 penalizes repeats)", 1.0, 2.0, 1.2)

if st.button("Generate Text"):
    with st.spinner("The CPU is processing the tokens..."):
        prompt_cleaned = clean_text(prompt_text)
        context = torch.tensor(encode(prompt_cleaned), dtype=torch.long, device=device).unsqueeze(0)
        
        generated_tensor = model.generate(context, max_new_tokens=tokens_to_generate, temperature=temperature, top_k=top_k if top_k > 0 else None, repetition_penalty=repetition_penalty)[0].tolist()
        result_text = decode(generated_tensor)
        
        st.subheader("Generated Output:")
        st.write(result_text)
