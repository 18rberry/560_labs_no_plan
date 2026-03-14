from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFacePipeline
from langchain_community.embeddings import SentenceTransformerEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import textwrap

# CONFIGURATION 
PDF_PATH = '/Users/ryantung/Downloads/Ads cookbook .pdf'
SBERT_MODEL = 'all-MiniLM-L6-v2'
LLM = 'google/flan-t5-small'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
K_RETRIEVE = 3


def conversation_chain(llm, vector_store):
    # Initialize model
    tokenizer = AutoTokenizer.from_pretrained(llm)
    model = AutoModelForSeq2SeqLM.from_pretrained(llm)

    # Build pipeline
    model_pipeline = pipeline(
        'text2text-generation',
        model=model,
        tokenizer=tokenizer,
        max_length=256,
        truncation=True,
        device=-1
    )

    # Apply LangChain wrapper
    llm_pipeline = HuggingFacePipeline(pipeline=model_pipeline)

    # Initialize memeory
    memory = ConversationBufferMemory(
        memory_key='chat_history',
        return_messages=True
    )

    # Build retriever
    retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': K_RETRIEVE})

    # Create chain
    conversation = ConversationalRetrievalChain.from_llm(
        llm=llm_pipeline,
        retriever=retriever,
        memory=memory,
        verbose=False
    )

    return conversation


def driver_function():
    # Load PDF into pages
    loader = PyPDFLoader(PDF_PATH)
    pages = loader.load()

    # Split text into chunks
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )

    chunks = text_splitter.split_documents(pages)

    # Create vector store
    embedding_model = SentenceTransformerEmbeddings(model_name=SBERT_MODEL)
    vector_store = FAISS.from_documents(chunks, embedding_model)

    # Create conversation chain
    conversation = conversation_chain(LLM, vector_store)

    # User interface loop
    print('\n' + '=' * 60)
    print('  ADS Textbook Chatbot')
    print('=' * 60)
    print("Ask questions about the ADS textbook. Type 'exit' to quit.")
    print('=' * 60 + '\n')

    while True:
        question = input('You: ')

        if question.lower() == 'exit':
            print('\nChatbot: Goodbye!\n')
            break

        response = conversation.invoke({'question': question})

        # Wrap long answers
        answer = textwrap.fill(response['answer'], width=60)

        print(f'\nChatBot:\n{answer}\n')
        print('=' * 60 + '\n')


if __name__ == '__main__':
    driver_function()


