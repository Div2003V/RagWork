import os
from dotenv import load_dotenv

load_dotenv()
# load_dotenv(dotenv_path=os.path.join(os.getcwd(), '.env'))

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA


llm = ChatOllama(model="llama3.2")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

def SELF_MADE_RAG_PROJECT():

    print("Loading documents...")
    loader = DirectoryLoader('./docs', glob="./*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    print("Creating Vector Database...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # vector_db.save_local("faiss_index")

    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever()
    )

    # query = "What is the main summary of these documents?"
    # response = rag_chain.invoke(query)
    
    # print("\n--- RESPONSE ---")
    # print(response["result"])

if __name__ == "__main__":
    # 'docs' folder with some files in it!

    # if not os.path.exists('docs'):
    #     os.makedirs('docs')
    SELF_MADE_RAG_PROJECT()