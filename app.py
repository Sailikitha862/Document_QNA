# streamlit_app.py

import streamlit as st
from docQA import load_pdf_and_initialize_retriever, get_bot_response



st.title("Document QA Chatbot")
st.markdown("Upload a PDF and ask questions!")

# Upload file
uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Load PDF and initialize retriever
if uploaded_file:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())
    load_pdf_and_initialize_retriever("temp.pdf")
    st.success("PDF loaded successfully! You can now ask questions.")

# Chat input
if uploaded_file:
    question = st.text_input("Ask your question:")
    if question:
        with st.spinner("Thinking..."):
            response = get_bot_response(question)
        st.markdown("**AI Response:**")
        st.write(response)


