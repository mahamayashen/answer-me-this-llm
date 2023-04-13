import streamlit as st
import pandas as pd
import os
import pinecone

from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)


pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment="northamerica-northeast1-gcp"  # next to api key in console
)
index_name = "querypdf"

def generate_response(prompt):
    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY)
    chain = load_qa_chain(llm, chain_type="stuff")
    docs = docsearch.similarity_search(prompt, include_metadata=True)
    message = chain.run(input_documents=docs, question=prompt)
    return message


st.title("AI reading Assistant: upload a PDF file and ask about it!")


uploaded_file = st.file_uploader("Choose a file")

if uploaded_file is not None:
    loader = UnstructuredPDFLoader(uploaded_file)
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name="querypdf")


prompt = st.text_input("Enter your question:", key='prompt')
if st.button("Submit", key='submit'):
  response = generate_response(prompt)
  st.success(response)