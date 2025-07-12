# PDF Chatbot with Google Gemini

A Streamlit-based chatbot that allows users to upload PDF documents and ask questions about their content using Google Gemini (Generative AI) for embeddings and chat completion.

## Features

- Upload and process PDF files
- Extract and split text into manageable chunks
- Generate vector embeddings using Google Gemini
- Store and search document chunks with FAISS
- Answer user questions about the uploaded document

## Requirements

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [pypdf](https://pypi.org/project/pypdf/)
- [langchain](https://python.langchain.com/)
- [langchain-google-genai](https://pypi.org/project/langchain-google-genai/)
- [faiss-cpu](https://pypi.org/project/faiss-cpu/)

## Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/rahul1257/ai-ml.git
   cd ai-ml
   
2. **Install the required packages:**
   ```bash
    pip install -r requirements.txt
    ```
3. **Set up Google Gemini API:**
4. Create a `.env` file in the root directory and add your Open API key:
   ```plaintext
   OPENAI_API_KEY=your_api_key_here
   ```
5. **Run the Streamlit app:**
   ```bash
    streamlit run chatbot.py
    ```