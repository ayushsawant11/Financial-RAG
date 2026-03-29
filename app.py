import streamlit as st
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1. Load API Keys from your .env file
load_dotenv()

# 2. Page Setup
st.set_page_config(page_title="Financial RAG", page_icon="📈")
st.title("📈 SEC 10-K Financial Assistant")

# 3. Initialize AI Tools (Cached so they only load once)
@st.cache_resource
def load_ai_tools():
    # Load the LLM
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    # Load the Embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Connect to the local database (Assuming it's stored in a folder called 'chroma_db')
    vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vectorstore.as_retriever()
    
    return llm, retriever

llm, retriever = load_ai_tools()

# 4. Create the RAG Prompt Template
system_prompt = (
    "You are a helpful financial assistant. Use the following retrieved context to answer the user's question. "
    "If you don't know the answer, just say that you don't know. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

# 5. Build the RAG Chain
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# 6. Streamlit Chat Interface Setup
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display previous chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 7. Chat Input & Processing
if user_input := st.chat_input("Ask a question about the financial documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Generate AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = rag_chain.invoke({"input": user_input})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"An error occurred: {e}")