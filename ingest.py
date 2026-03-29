from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import os

print("1. Loading PDF...")
loader = PyPDFLoader("Tesla10-k.pdf")
docs = loader.load()

print("2. Chopping text into chunks...")
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)

print("3. Loading Embedding Model...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

print("4. Saving to Chroma Database...")
vectorstore = Chroma.from_documents(
    documents=splits,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

print("✅ SUCCESS! Your database is built. You can now run Streamlit.")