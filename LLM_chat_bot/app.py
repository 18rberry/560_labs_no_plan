import os
import threading
from flask import Flask, request, jsonify, render_template

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

from open_source_chat import conversation_chain
import openai_chat

app = Flask(__name__)
UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

SBERT_MODEL = 'all-MiniLM-L6-v2'
LLM_MODEL = 'google/flan-t5-small'
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

app_state = {
    'chain': None,
    'pdf_names': [],
    'status': 'idle',
    'status_msg': 'Upload PDFs to begin.',
    'backend': 'opensource',
}
state_lock = threading.Lock()


def process_pdfs(file_paths, backend):
    with state_lock:
        app_state['status'] = 'processing'
        app_state['status_msg'] = 'Loading and splitting PDFs...'

    try:
        pages = []
        for path in file_paths:
            loader = PyPDFLoader(path)
            pages.extend(loader.load())

        splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
        )
        chunks = splitter.split_documents(pages)

        if backend == 'openai':
            with state_lock:
                app_state['status_msg'] = 'Building OpenAI vector store...'
            vector_store = openai_chat.create_vector_store(chunks)
            llm = openai_chat.get_llm()
            chain = openai_chat.create_conversation_chain(llm, vector_store)
        else:
            with state_lock:
                app_state['status_msg'] = 'Loading HuggingFace model (this may take a minute)...'
            embedding_model = SentenceTransformerEmbeddings(model_name=SBERT_MODEL)
            vector_store = FAISS.from_documents(chunks, embedding_model)
            chain = conversation_chain(LLM_MODEL, vector_store)

        with state_lock:
            app_state['chain'] = chain
            app_state['status'] = 'ready'
            app_state['status_msg'] = 'Ready to chat!'

    except Exception as e:
        with state_lock:
            app_state['status'] = 'error'
            app_state['status_msg'] = f'Error: {e}'


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload():
    files = request.files.getlist('files')
    backend = request.form.get('backend', 'opensource')

    if not files or all(f.filename == '' for f in files):
        return jsonify({'error': 'No files provided'}), 400

    saved_paths = []
    pdf_names = []
    for f in files:
        if f.filename == '':
            continue
        path = os.path.join(UPLOAD_FOLDER, f.filename)
        f.save(path)
        saved_paths.append(path)
        pdf_names.append(f.filename)

    with state_lock:
        app_state['chain'] = None
        app_state['pdf_names'] = pdf_names
        app_state['backend'] = backend
        app_state['status'] = 'processing'
        app_state['status_msg'] = 'Starting processing...'

    thread = threading.Thread(target=process_pdfs, args=(saved_paths, backend), daemon=True)
    thread.start()

    return jsonify({'message': 'Processing started', 'pdf_names': pdf_names}), 202


@app.route('/status', methods=['GET'])
def status():
    with state_lock:
        return jsonify({
            'status': app_state['status'],
            'message': app_state['status_msg'],
            'backend': app_state['backend'],
            'pdf_names': app_state['pdf_names'],
        })


@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    message = (data or {}).get('message', '').strip()
    if not message:
        return jsonify({'error': 'Empty message'}), 400

    with state_lock:
        chain = app_state['chain']

    if chain is None:
        return jsonify({'error': 'No PDF loaded yet'}), 400

    try:
        response = chain.invoke({'question': message})
        answer = response.get('answer', str(response))
        return jsonify({'answer': answer})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/reset', methods=['POST'])
def reset():
    with state_lock:
        app_state['chain'] = None
        app_state['pdf_names'] = []
        app_state['status'] = 'idle'
        app_state['status_msg'] = 'Upload PDFs to begin.'
        app_state['backend'] = 'opensource'
    return jsonify({'message': 'Reset successful'})


if __name__ == '__main__':
    app.run(debug=True, port=5000)
