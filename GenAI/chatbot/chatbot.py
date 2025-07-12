import streamlit as st
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.chat_models import ChatOpenAI


# Set your OpenAI API key here
OPENAI_API_KEY = ""

#upload pdf file
st.header("My First Chatbot")

with st.sidebar:
    st.title("Your Documents")
    file = st.file_uploader("Upload a PDF file", type=["pdf"])

#Extract the text
if file is not None:
  pdf_reader = PdfReader(file)
  text = ""
  for page in pdf_reader.pages:
    text += page.extract_text()
    # st.write(text)

  text_splitter = RecursiveCharacterTextSplitter(
      separators = "\n",
      chunk_size = 1000,
      chunk_overlap=150,
      length_function=len
  )
  chunks = text_splitter.split_text(text)
  # st.write(chunks)

  # generating  embeddings
  embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

  # Creating Vector store - FAISS (Facebook AI Semantics Search and Similarity)
  vector_store = FAISS.from_texts(chunks, embeddings)

  # get user question
  user_question = st.text_input("Ask a question about the document", key="user_question")

  # do similarity search
  if user_question:
      # get top 5 similar chunks
      match = vector_store.similarity_search(user_question, k=5)
      # st.write(match)

      llm = ChatOpenAI(
          open_api_key = OPENAI_API_KEY,
          temperature = 0.0,
          max_tokens = 1000,
          model = "gpt-3.5-turbo"  # or "gpt-4" if you have access
      )

      # output results
      #take the question, get relevant documents, and answer the question
      chain = load_qa_chain(llm, chain_type="stuff")
      response = chain.run(input_documents = match, question = user_question)
      st.write(response)
