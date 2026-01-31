import streamlit as st
import numpy as np
from .classif import *

@st.cache_data(show_spinner=False)
def shap_explain(text_input, predicted_class):
    class_idx = st.session_state['classif_model'].config.label2id[predicted_class]
    shap_values = st.session_state['shap_explainer']([text_input])
    values = shap_values.values[0].tolist()[1:-1]
    tokens = st.session_state['classif_tokenizer'].tokenize(text_input)
    shap_exp = []
    for i, (token, value) in enumerate(zip(tokens, values)):
        shap_exp.append([i, token, value[class_idx]])
    return shap_exp

@st.cache_data(show_spinner=False)
def lime_explain(text_input, predicted_class, random_state=0):
    class_idx = st.session_state['classif_model'].config.label2id[predicted_class]

    tokens = st.session_state['classif_tokenizer'].tokenize(text_input)
    reconstructed_text = st.session_state['classif_tokenizer'].decode(st.session_state['classif_tokenizer'].convert_tokens_to_ids(tokens))

    # these two lines are used to make sure random state works
    st.session_state['lime_explainer'].random_state = np.random.RandomState(random_state)
    st.session_state['lime_explainer'].base.random_state = st.session_state['lime_explainer'].random_state

    num_samples = 50
    exp = st.session_state['lime_explainer'].explain_instance(reconstructed_text, predictor, num_samples=num_samples, labels=[class_idx])

    lime_exp = exp.as_list(st.session_state['classif_model'].config.label2id[predicted_class])
    lime_exp_dict = dict(lime_exp)
    lime_exp = [[i, token, lime_exp_dict.get(token, lime_exp_dict.get(token[2:], 0))] for i, token in enumerate(tokens)]
    return lime_exp

def token_to_word_explanation(token_explanation, aggregation="sum"):
    words = []
    current_word = ""
    current_score = 0.0

    j = 0
    for _, token, score in token_explanation:
        if token.startswith("##"):
            # merge subword with previous token
            current_word += token[2:]
            if aggregation == "sum":
                current_score += score
            elif aggregation == "mean":
                current_score = (current_score + score) / 2
        else:
            # save previous word if exists
            if current_word:
                words.append([j, current_word, current_score])
                j += 1
            # start new word
            current_word = token
            current_score = score

    # append last word
    if current_word:
        words.append([j, current_word, current_score])
        j += 1

    return words

def top_k_words(word_explanation, k=3):
    sorted_words = sorted(word_explanation, key=lambda x: x[2], reverse=True)
    filtered_exp = sorted_words[:k]
    # return filtered_exp
    return [w for i, w, s in filtered_exp]