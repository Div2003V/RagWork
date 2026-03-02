import os
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

from dotenv import load_dotenv

load_dotenv()


DB_PATH = "faiss_index"
MODEL_NAME = "llama3.2"
EMBEDDING_MODEL = "nomic-embed-text"

def start_history_aware_chat():
    print("\n" + "="*50)
    print("HISTORY-AWARE LOCAL RAG CHAT")
    print("="*50)

    llm = ChatOllama(model=MODEL_NAME)
    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    
    if not os.path.exists(DB_PATH):
        print("Error: Vector DB not found. Run your retrieval_pipeline first!")
        return
    
    vector_db = FAISS.load_local(DB_PATH, embeddings, allow_dangerous_deserialization=True)
    retriever = vector_db.as_retriever()

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    system_prompt = (
        "You are a strict document-based assistant. "
        "ONLY use the following pieces of retrieved context to answer the question. "
        "If the answer is not contained within the context, specifically state that "
        "the information is not available in the provided documents. "
        "DO NOT use your own external knowledge or provide facts from outside the context."
        "\n\n"
        "{context}"
    )
    
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    chat_history = [] 
    
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in ['exit', 'quit']: break

        print("\nThinking...")
        
        response = rag_chain.invoke({
            "input": user_input, 
            "chat_history": chat_history
        })
        
        print(f"\nAI: {response['answer']}")

        chat_history.extend([
            HumanMessage(content=user_input),
            AIMessage(content=response["answer"]),
        ])
        
        if len(chat_history) > 6:
            chat_history = chat_history[-6:]

if __name__ == "__main__":
    start_history_aware_chat()