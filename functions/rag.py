from sklearn.metrics.pairwise import cosine_similarity
from google.genai import types
import streamlit as st

@st.cache_data(show_spinner=False)
def retrieve(query, law_category=None, k=3):
    query_embedding = st.session_state['embed_model'].encode(f"query: {query}")
    if law_category:
        filtered_articles = st.session_state['articles_emb'][st.session_state['articles_emb'].merged_category == law_category]
    else:
        filtered_articles = st.session_state['articles_emb'].copy()
    similarities = cosine_similarity([query_embedding], list(filtered_articles["article_text_emb"]))[0]

    filtered_articles.loc[:, "score"] = similarities
    top_docs = filtered_articles.sort_values("score", ascending=False).head(k)
    return top_docs

@st.cache_data(show_spinner=False)
def generate_answer(query, context):
    context = "\n\n".join(context["article_text"].tolist())
    prompt = f"""
    أجب عن السؤال باستخدام المعلومات التالية فقط.
    إذا لم تتضمن المعلومات المقدمة إجابة السؤال، فاذكر فقط أنه لا توجد معلومات كافية للإجابة.

    المعلومات:
    {context}

    السؤال:
    {query}

    الإجابة:
    """
    contents = [
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)]),
    ]
    response = st.session_state['gemini_client'].models.generate_content(
        model=st.session_state['gemini_model'], contents=contents,
        config=st.session_state['generate_content_config'],
    )
    return response.candidates[0].content.parts[0].text