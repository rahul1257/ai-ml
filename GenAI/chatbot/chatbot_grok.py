import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings

# Set your Groq API key here
GROQ_API_KEY = ""  #location for key should be at https://console.groq.com/keys

st.header("Groq PDF Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file", type=["pdf"])

if file is not None:
    pdf_reader = PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # st.write("extracted text: ", text[:5000])

    text_splitter = RecursiveCharacterTextSplitter(
        separators="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    # st.write("chunks", chunks)

    # generating embeddings (free, local)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # or "cuda" if GPU available
    )
    # st.write("embeddings", embeddings)

    # Creating Vector store - FAISS (Facebook AI Semantics Search and Similarity)
    vector_store = FAISS.from_texts(chunks, embeddings)

    user_question = st.text_input("Ask a question about the document", key="user_question")

    if user_question:
        match = vector_store.similarity_search(user_question, k=5)
        # st.write("matched chunks", match)

        # Usage:
        llm = ChatGroq(
            groq_api_key=GROQ_API_KEY,
            temperature=0.0,
            max_tokens=1000,
            model="llama-3.3-70b-versatile"
        )

        chain = load_qa_chain(llm, chain_type="stuff")
        response = chain.run(input_documents=match, question=user_question)
        st.write(response)
