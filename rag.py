import os
import tempfile
from dotenv import load_dotenv

from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("Missing GROQ_API_KEY in environment variables.")

llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def query_from_urls(urls, question):
    # Load documents
    loader = WebBaseLoader(urls)
    docs = loader.load()

    # Chunk the documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = splitter.split_documents(docs)

    # Use Chroma in a temp directory
    with tempfile.TemporaryDirectory() as tmpdirname:
        vectordb = Chroma.from_documents(documents, embeddings, persist_directory=tmpdirname)

        # Build retrieval chain
        retriever = vectordb.as_retriever()
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)

        # Ask the question
        result = qa_chain.invoke({"query": question})
        return result['result'], urls
