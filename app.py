import streamlit as st
from PyPDF2 import PdfReader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter  # Added this import
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_embeddings(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return embeddings

def get_conversational_chain(vectorstore):
    llm = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversational_chain

def handle_userinput(user_question):
    response = st.session_state.conversational_chain({"question": user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(':man_in_tuxedo:', message.content)
        else:
            st.write(':robot_face:', message.content)

def main():
    st.set_page_config(page_title="Chat with your Documents", page_icon=":books:")
    st.title("AI Chat-Bot for Your Documents :books:")

    if "conversational_chain" not in st.session_state:
        st.session_state.conversational_chain = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    user_input = st.text_input("Upload files, Hit Submit button, then ask the questions.:blue[- Mythresh N]:sunglasses:")
    if user_input and st.session_state.conversational_chain:
        handle_userinput(user_input)

    with st.sidebar:
        pdf_docs = st.file_uploader("Please Upload your documents here", accept_multiple_files=True)
        if st.button("Submit"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                chunked_array = get_text_chunks(raw_text)
                embeddings = get_embeddings(chunked_array)
                vectorstore = FAISS.from_texts(chunked_array, embeddings)
                st.session_state.conversational_chain = get_conversational_chain(vectorstore)

if __name__ == "__main__":
    main()


