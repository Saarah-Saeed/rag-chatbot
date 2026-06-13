import streamlit as st

st.set_page_config(page_title="PDF RAG Chatbot")

st.title("📚 PDF RAG Chatbot")

question = st.text_input("Ask a question about the PDF")

if question:
    st.write("Answer will appear here")