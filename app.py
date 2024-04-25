import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
import time
from gtts import gTTS
import pickle
import hashlib
from pydub import AudioSegment
from pydub.playback import play


os.environ["GROQ_API_KEY"]= st.secrets["GROQ_API_KEY"]
os.environ["HF_API_KEY"]= st.secrets["HF_API_KEY"]
os.environ["LANGCHAIN_TRACING_V2"]= st.secrets["LANGCHAIN_TRACING_V2"]
os.environ["LANGCHAIN_ENDPOINT"]= st.secrets["LANGCHAIN_ENDPOINT"]
os.environ["LANGCHAIN_API_KEY"]= st.secrets["LANGCHAIN_API_KEY"]
os.environ["LANGCHAIN_PROJECT"]= st.secrets["LANGCHAIN_PROJECT"]


# Set the path to your vector store directory
def load_vectors():
    VECTOR_STORE_DIR = "./vector_store"
    CATALOG_DIR = "./Content"
    # Create the vector store directory if it doesn't exist
    os.makedirs(VECTOR_STORE_DIR, exist_ok=True)
    # Calculate the hash of the catalog directory
    catalog_hash = hashlib.sha256(str([os.path.join(CATALOG_DIR, f) for f in os.listdir(CATALOG_DIR)]).encode()).hexdigest()
    # Check if the vector store already exists
    vector_store_path = os.path.join(VECTOR_STORE_DIR, "vector_store.pkl")


    def load_multiple_files(directory_path):
        documents = []
        urls = ['https://github.com/RahulManocha21?tab=repositories',
                'https://rahulmanocha.vercel.app/',
                'https://www.linkedin.com/in/manocharahul/'
                ]
        for url in urls:
            webloader = WebBaseLoader(url)
            documents.extend(webloader.load())
        for filename in os.listdir(directory_path):
            if filename.endswith(".csv"):
                file_path = os.path.join(directory_path, filename)
                csvloader = CSVLoader(file_path)
                documents.extend(csvloader.load())
            elif filename.endswith(".pdf"):
                file_path = os.path.join(directory_path, filename)
                pdfloader = PyPDFLoader(file_path)
                documents.extend(pdfloader.load())
        return documents

    if os.path.exists(vector_store_path):
        # Load the existing vector store and catalog hash
        with open(vector_store_path, "rb") as f:
            vectors, stored_catalog_hash = pickle.load(f)
        # Check if the catalog has changed
        if catalog_hash != stored_catalog_hash:
            # Regenerate the vector store
            with st.spinner('Updating vector store...'):
                embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5", encode_kwargs={'normalize_embeddings': True})
                docs = load_multiple_files(CATALOG_DIR)
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                final_documents = text_splitter.split_documents(docs)
                vectors = FAISS.from_documents(final_documents, embeddings)

            # Save the new vector store and catalog hash
            with open(vector_store_path, "wb") as f:
                pickle.dump((vectors, catalog_hash), f)
    else:
        # Create a new vector store
        with st.spinner('Updating vector store...'):
            embeddings = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-small-en-v1.5", encode_kwargs={'normalize_embeddings': True})
            docs = load_multiple_files(CATALOG_DIR)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            final_documents = text_splitter.split_documents(docs)
            vectors = FAISS.from_documents(final_documents, embeddings)

        # Save the new vector store and catalog hash
        with open(vector_store_path, "wb") as f:
            pickle.dump((vectors, catalog_hash), f)

    return vectors

def clear():
            st.session_state.messages = []
            st.session_state.messages.append({"role": "assistant", "content": "🙋🏼Welcome! Allow me to introduce Newton, my AI Assistant. He's adept at giving you a quick rundown about me. Feel free to engage in a pleasant conversation with him. Enjoy your time here!👋"})
            
def text_to_speech(text, lang='en'):
    filename='output.mp3'
    tts = gTTS(text=text, lang=lang, tld='com')
    tts.save(filename)
    sound = AudioSegment.from_file("output.mp3", format="mp3")
    play(sound)
    os.remove(filename)
   

st.set_page_config(page_title='Chat with me', page_icon='📝', layout='wide', initial_sidebar_state="collapsed")
try:
    ######Initialization 
    llm = ChatGroq(groq_api_key=os.environ['GROQ_API_KEY'], model_name="llama3-70b-8192",
                   temperature=1)
    prompt = ChatPromptTemplate.from_template(
        """
        You name is Newton Assistant of Mr. Rahul Manocha, here users ask you about rahul manocha.
        if input is any abusive word or get lost, or any kind of words that can be not a good behave, only then warn the user not to use that words.
        Do not introduce you again and again in every response. 
        If the question is general statements like thanks, no thanks, sorry, good do not provide the response on base of context.
        <context>
        {context}
        <context>
        Questions:{input}
        Conversation History: {history}
        """
    )
    vectors = load_vectors()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    # qa = RetrievalQA.from_chain_type(document_chain, retriever)
    if st.button("Clear Chat"):
            clear()
    if "messages" not in st.session_state:
            clear()
    chatcontainer = st.container(border=True, height=330)
    
    for message in st.session_state.messages:
                if message["role"] == 'user':
                    chatcontainer.chat_message(message["role"], avatar='🧑‍🦳')
                    chatcontainer.write(message["content"])
                else:
                    chatcontainer.chat_message(message["role"], avatar='👤')
                    chatcontainer.write(message["content"])

    if prompt := st.chat_input("Ask your Questions"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chatcontainer.chat_message("user", avatar='🧑‍🦳'):
            chatcontainer.write(prompt)

        with chatcontainer.chat_message("assistant", avatar='👤'):
            response  =  retrieval_chain.invoke({'input':prompt, 'history': st.session_state.messages})
            chatcontainer.write(response['answer'])
            
        st.session_state.messages.append({"role": "assistant", "content": response['answer']})

except Exception as e:
    # st.error(e)
    st.warning("I am down, my boss is working on me to make me more smarter than yesterday")
    