from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

load_dotenv()


# CONSTANTS

CHUNK_SIZE = 1000
EMBEDDING_MODEL = 'Alibaba-NLP/gte-base-en-v1.5'
VECTORSTORE_DIR = Path(__file__).parent/'resources/vectorstore'
COLLECTION_NAME = 'real_state'

llm = None
vectore_store = None

def initialize_components():
    global llm, vectore_store

    if llm is None:
        llm = ChatGroq(model='llama-3.3-70b-versatile', temperature=0.9, max_tokens=500)

    if vectore_store is None:
        ef = HuggingFaceEmbeddings(
            model_name = EMBEDDING_MODEL,
            model_kwargs = {'trust_remote_code': True}

        )

        vectore_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )

def process_urls(urls): 

    print("Initialize components")

    initialize_components()

    vectore_store.reset_collection()

    print("Load data")

    loader = UnstructuredURLLoader(urls=urls)
    data = loader.load()

    print("Split Text")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = CHUNK_SIZE,
        separators=['\n\n','\n',' ']
    )
    docs = text_splitter.split_documents(data)

    print("Add docs to vector db")

    uuids = [str(uuid4()) for _ in range(len(docs))]

    vectore_store.add_documents(docs, ids=uuids)

if __name__ == "__main__":
    urls = [
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]

    process_urls(urls)

    results = vectore_store.similarity_search(
        '30 year mortage rate',
        k=2
    )

    print(results)