import os
import streamlit as st
import pdfplumber
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from tempfile import NamedTemporaryFile

# Set your OpenAI API Key
openai_api_key = st.secrets["openai_api_key"] if "openai_api_key" in st.secrets else os.getenv("OPENAI_API_KEY")

st.title("📄 Resume Chatbot")
st.write("Ask questions about your uploaded resume!")

uploaded_file = st.file_uploader("Upload your Resume (PDF)", type="pdf")

if uploaded_file and openai_api_key:
    with pdfplumber.open(uploaded_file) as pdf:
        text = ""
        for page in pdf.pages:
            text += page.extract_text() or ""

    # Split text into chunks
    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100)
    chunks = splitter.split_text(text)

    # Create vector store
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    db = Chroma.from_texts(chunks, embeddings)

    # Retrieval-based QA
    llm = ChatOpenAI(openai_api_key=openai_api_key)
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=db.as_retriever())

    question = st.text_input("Ask a question about your resume:")

    if question:
        result = qa.run(question)
        st.write("🧠 Answer:", result)

elif not openai_api_key:
    st.warning("Please set your OpenAI API key.")
