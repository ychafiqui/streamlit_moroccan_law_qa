import streamlit as st
import pandas as pd
from functions.classif import *
from functions.rag import *
from functions.xai import *
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import shap
from lime import lime_text
from sentence_transformers import SentenceTransformer
from google import genai
from google.genai import types
import torch
import time

st.set_page_config(
    page_title="Moroccan Legal Q&A",
    page_icon="⚖️", layout="wide",
    initial_sidebar_state="expanded"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_index = 0 if torch.cuda.is_available() else -1  # -1 means CPU for HF pipeline

@st.cache_resource(show_spinner="Loading classification model...", show_time=True)
def load_classification_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
    classif_pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=device_index)
    return tokenizer, model, classif_pipe

# classification model
if 'classif_model' not in st.session_state:
    tokenizer, model, classif_pipe = load_classification_model("ychafiqui/moroccan_law_qst_classif_arabert2")
    st.session_state['classif_tokenizer'] = tokenizer
    st.session_state['classif_model'] = model
    st.session_state['classif_pipe'] = classif_pipe

@st.cache_resource(show_spinner="Loading embedding model...", show_time=True)
def load_embedding_model(model_name):
    embed_model = SentenceTransformer(model_name, device=device)
    return embed_model

# embedding model
if 'embed_model' not in st.session_state:
    st.session_state['embed_model'] = load_embedding_model("mhaseeb1604/bge-m3-law")

# generative AI model
if 'gemini_client' not in st.session_state:
    st.session_state['gemini_client'] = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
    st.session_state['gemini_model'] = "gemini-flash-lite-latest"
    st.session_state['generate_content_config'] = types.GenerateContentConfig(thinking_config=types.ThinkingConfig(thinking_budget=0))

@st.cache_data(show_spinner="Loading legal articles data...", show_time=True)
def load_articles_embeddings():
    articles_emb = pd.read_parquet("articles_emb.parquet")
    # articles_emb["article_text_emb"] = articles_emb["article_text_emb"].apply(lambda x: eval(x))
    return articles_emb

# load articles embeddings
if 'articles_emb' not in st.session_state:
    st.session_state['articles_emb'] = load_articles_embeddings()

@st.cache_resource(show_spinner="Loading SHAP explainer...", show_time=True)
def load_shap_explainer():
    max_evals = 50
    random_state = 0
    explainer = shap.Explainer(st.session_state['classif_pipe'], 
        st.session_state['classif_tokenizer'], algorithm="partition", max_evals=max_evals, 
        seed=random_state, output_names=list(st.session_state['classif_model'].config.id2label.values()))
    return explainer

@st.cache_resource(show_spinner="Loading LIME explainer...", show_time=True)
def load_lime_explainer():
    random_state = 0
    explainer = lime_text.LimeTextExplainer(
        class_names=list(st.session_state['classif_model'].config.label2id.keys()), 
        split_expression=tokenize, feature_selection='none', random_state=random_state)
    return explainer

# load explainers
if 'shap_explainer' not in st.session_state:
    st.session_state['shap_explainer'] = load_shap_explainer()
if 'lime_explainer' not in st.session_state:
    st.session_state['lime_explainer'] = load_lime_explainer()

user_question = st.text_input("Ask your legal question here:")
# كيف يتم إيقاف سحب رخصة السياقة المقررة من قبل الإدارة؟
# ما هي الجهة المختصة التي يمكنني تقديم طلب إثبات الزواج أو انحلاله إليها؟
answer_btn = st.button("Answer")

if answer_btn:
    if user_question.strip():
        start_time = time.time()

        with st.spinner("Classifying the legal question...", show_time=True):
            law_category = predict_law_category(user_question)
            probas = predict_proba(user_question)
        with st.expander("**Legal Category Classification**"):
            st.markdown(f'Predicted legal category: <span lang="ar">**{law_category}**</span>', unsafe_allow_html=True)
            proba_df = pd.DataFrame.from_dict(probas, orient='index', columns=['Probability'])
            proba_df = proba_df.sort_values(by='Probability', ascending=False)
            st.bar_chart(proba_df, color="rgba(255, 0, 0, 0.6)", horizontal=True, sort=False)

        xai_placeholder = st.empty()
        xai_container = st.container()

        with st.spinner("Retrieving relevant legal articles...", show_time=True):
            context = retrieve(user_question, law_category=law_category, k=5)
        with st.expander("**Legal Articles Retrieval**"):
            for idx, row in context.iterrows():
                with st.expander(f"**{row['category']} - {row['file_name']} - {row['article_number']}** ({100 * row['score']:.0f}% similarity)"):
                    st.markdown(f'<span lang="ar">{row["article_text"]}</span>', unsafe_allow_html=True)

        with st.spinner("Generating answer...", show_time=True):
            answer = generate_answer(user_question, context)
        with st.expander("**Answer Generation**", expanded=True):
            st.write(answer)

        with xai_placeholder.spinner("Generating explanations...", show_time=True):
            shap_exp = shap_explain(user_question, law_category)
            lime_exp = lime_explain(user_question, law_category)

        shap_tab, lime_tab = xai_container.expander("**Classification explanation**").tabs(["SHAP", "LIME"])
        with shap_tab:
            shap_word_exp = token_to_word_explanation(shap_exp, aggregation="sum")
            shap_top_words = top_k_words(shap_word_exp, k=3)
            template = "Your question was classified to the <span lang='ar'>**{}**</span> legal category because of the presence of the following **top-{}** words: <span lang='ar'>**{}**</span>"
            st.markdown(template.format(law_category, len(shap_top_words), " - ".join([word for word in shap_top_words])), unsafe_allow_html=True)

            st.write("**Word importance:**")
            shap_sentence = ""
            for _, word, score in shap_word_exp:
                shap_sentence += (
                    f"<span style='background-color:rgba(255, 0, 0, {score}); "
                    f"padding:2px 2px; border-radius:4px;'>"
                    f"<span lang='ar'>{word}</span></span> "
                )
            st.markdown(shap_sentence, unsafe_allow_html=True)
        with lime_tab:
            lime_word_exp = token_to_word_explanation(lime_exp, aggregation="mean")
            lime_top_words = top_k_words(lime_word_exp, k=3)
            template = "Your question was classified to the legal category: <span lang='ar'>**{}**</span> because of the presence of the following **top-{}** words: <span lang='ar'>**{}**</span>"
            st.markdown(template.format(law_category, len(lime_top_words), " - ".join([word for word in lime_top_words])), unsafe_allow_html=True)

            st.write("**Word importance:**")
            lime_sentence = ""
            for _, word, score in lime_word_exp:
                lime_sentence += (
                    f"<span style='background-color:rgba(255, 0, 0, {score}); "
                    f"padding:2px 2px; border-radius:4px;'>"
                    f"<span lang='ar'>{word}</span></span> "
                )
            st.markdown(lime_sentence, unsafe_allow_html=True)

        total_time = time.time() - start_time
        st.write("**Total time:** {:.2f} seconds.".format(total_time))
    else:
        st.warning("Please enter a valid question before submitting.")