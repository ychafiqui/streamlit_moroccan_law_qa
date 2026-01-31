import streamlit as st

st.set_page_config(
    page_title="Moroccan Legal Chatbot",
    page_icon="⚖️", layout="wide",
    initial_sidebar_state="expanded"
)

# Load custom arabic fonts
st.markdown('<style>@import url("/static/fonts.css");</style>', unsafe_allow_html=True)

st.title("⚖️ Moroccan Legal Q&A")
st.markdown(
    """
    This application is a Moroccan Legal Q&A that leverages advanced NLP techniques to assist users in understanding Moroccan law.

    **Features:**
    - Ask legal questions and receive informed responses based on Moroccan law
    - Understand the reasoning behind the system's answers through XAI explanations
    - Access a curated database of Moroccan legal documents (will be available soon)
    - User-friendly interface for seamless interaction
    - Continuous updates with the latest legal information
    
    **Technologies Used:**
    - Pre-trained and fine-tuned language models for Moroccan legal texts classification
    - Explainable AI (XAI) methods to provide transparency in model predictions
    - Retrieval-Augmented Generation (RAG) for enhanced information retrieval

    ***Disclaimer :** This chatbot is intended for informational purposes only and should not be considered as legal advice. Always consult a qualified legal professional for specific legal matters.*
    
    Developed by **Youssef Chafiqui**
    """
)