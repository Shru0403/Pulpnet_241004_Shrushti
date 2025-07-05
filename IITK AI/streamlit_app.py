# app.py

import json
import torch
import streamlit as st
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# loading dataset and presaved embeddings
@st.cache_resource
def load_data():
    with open("dataset.json", "r", encoding="utf-8") as f:
        dataset = json.load(f)
    chunk_embeddings = torch.load("chunk_embeddings.pt")
    return dataset, chunk_embeddings

# load models
@st.cache_resource
def load_models():
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    qa_model = pipeline(
        "question-answering",
        model="distilbert-base-uncased-distilled-squad",
        tokenizer="distilbert-base-uncased-distilled-squad",
        device=-1  # CPU
    )
    return embedder, qa_model

dataset, chunk_embeddings = load_data()
embedder, qa_model = load_models()

# retrieval
def get_top_k_chunks(question, k=3):
    q_embed = embedder.encode(question, convert_to_tensor=True)
    scores = util.cos_sim(q_embed, chunk_embeddings)[0]
    top_indices = torch.topk(scores, k=k).indices
    return [dataset[i] for i in top_indices if dataset[i].strip()]

# answering
def answer_question(question, k=3):
    context_chunks = get_top_k_chunks(question, k=k)

    if not context_chunks:
        return "Sorry, I couldn't find any relevant information."

    context = " ".join(context_chunks)

    result = qa_model({
        "question": question,
        "context": context
    })

    return result["answer"]

# streamlit interface
st.set_page_config(page_title="IITK Chatbot", layout="centered")
st.title("🤖 IIT Kanpur Chatbot")
st.write("Ask me anything about IITK")

query = st.text_input("Enter your question:")

if query:
    with st.spinner("Thinking..."):
        try:
            answer = answer_question(query)
            st.markdown("### 🔹 Answer")
            st.success(answer)
        except Exception as e:
            st.error(f"Something went wrong: {e}")
