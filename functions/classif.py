import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

@st.cache_data(show_spinner=False)
def predict_law_category(text_input):
    return st.session_state['classif_pipe'](text_input)[0]["label"]

@st.cache_data(show_spinner=False)
def tokenize(text):
    tokens = st.session_state['classif_tokenizer'].tokenize(text)
    for i, token in enumerate(tokens):
        if token.startswith("##"):
            tokens[i] = tokens[i][2:]
    return tokens

@st.cache_data(show_spinner=False)
def predictor(texts):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inputs = st.session_state['classif_tokenizer'](texts, return_tensors="pt", padding=True).to(device)
    outputs = st.session_state['classif_model'](**inputs)
    probas = F.softmax(outputs.logits, dim=1).detach().cpu().numpy()
    return probas

@st.cache_data(show_spinner=False)
def predict_proba(text):
    probas = predictor([text])
    return {st.session_state['classif_model'].config.id2label[i]: float(probas[0][i]) for i in range(len(probas[0]))}