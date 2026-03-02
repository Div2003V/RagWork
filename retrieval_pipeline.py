import os
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from dotenv import load_dotenv

load_dotenv()

DOCS_PATH = "./docs"
DB_PATH = "faiss_index"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

def retrieval_pipeline():
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

    loader_mapping = {
        ".pdf": PyPDFLoader,
        ".txt": TextLoader,
        ".docx": Docx2txtLoader,
    }

    print(f"--- Loading documents ---")
    
    documents = []
    for ext, loader_cls in loader_mapping.items():
        loader = DirectoryLoader(DOCS_PATH, glob=f"**/*{ext}", loader_cls=loader_cls)
        documents.extend(loader.load())

    if not documents:
        print("No documents found! Please add files to the /docs folder.")
        return

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    print(f"--- Creating Vector Store for {len(chunks)} chunks ---")
    vector_db = FAISS.from_documents(chunks, embeddings)
    vector_db.save_local(DB_PATH)

    llm = ChatOllama(model=MODEL_NAME)
    
    template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    {context}
    
    Question: {question}
    Helpful Answer:"""
    
    QA_CHAIN_PROMPT = PromptTemplate.from_template(template)

    rag_pipeline = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", 
        retriever=vector_db.as_retriever(search_kwargs={"k": 3}), 
        chain_type_kwargs={"prompt": QA_CHAIN_PROMPT}
    )
    
    return rag_pipeline

if __name__ == "__main__":
    # if not os.path.exists(DOCS_PATH):
    #     os.makedirs(DOCS_PATH)
    #     print(f"Created {DOCS_PATH} folder. Add your files there and run again.")
    # else:
        pipeline = retrieval_pipeline()
        # if pipeline:
        #     query = input("\nAsk a question about your documents: ")
        #     result = pipeline.invoke(query)
        #     print("\n--- LLM RESPONSE ---")
        #     print(result["result"])