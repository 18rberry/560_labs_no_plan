import os
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), "riya.env"))
open_ai_key = os.getenv("NEW_OPENAI_API_KEY")

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def create_vector_store(chunks):
    embedding_model = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=open_ai_key
    )
    vector_store = FAISS.from_documents(chunks, embedding_model)
    return vector_store


def create_conversation_chain(llm, vector_store):
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(
            search_type="similarity", search_kwargs={"k": 6}
        ),
        memory=memory,
        verbose=False
    )
    return conversation


def get_llm():
    return ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=open_ai_key)
