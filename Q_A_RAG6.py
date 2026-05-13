
# ---------------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# ---------------------------------------------------------

# Streamlit for web UI
import streamlit as st

# OS module for file handling
import os

# PyPDF2 for reading PDF files
import PyPDF2

# YAML reader
import yaml

# Temporary file handling
import tempfile

# Recursive text splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter

# FAISS vector database
from langchain_community.vectorstores import FAISS

# HuggingFace embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Ollama local LLM
from langchain_ollama import OllamaLLM

# Conversational Retrieval Chain
from langchain_classic.chains import ConversationalRetrievalChain

# Conversation memory
from langchain_classic.memory import ConversationBufferMemory


# ---------------------------------------------------------
# LOAD YAML CONFIG FILE
# ---------------------------------------------------------

with open("config.yaml", "r") as file:
    config = yaml.safe_load(file)


# ---------------------------------------------------------
# READ VALUES FROM YAML FILE
# ---------------------------------------------------------

faiss_index_path = config["faiss_index_path"]
ollama_model = config["ollama_model"]
embedding_model = config["embedding_model"]
chunk_size = config["chunk_size"]
chunk_overlap = config["chunk_overlap"]


# ---------------------------------------------------------
# STREAMLIT PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="Advanced Local RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Advanced Local RAG PDF Chatbot")


# ---------------------------------------------------------
# PDF UPLOAD SECTION
# ---------------------------------------------------------

uploaded_files = st.file_uploader(
    "Upload PDF Files",
    type="pdf",
    accept_multiple_files=True
)


# ---------------------------------------------------------
# PROCESS PDF FILES
# ---------------------------------------------------------

if uploaded_files:

    raw_text = ""

    # Read uploaded PDFs
    for uploaded_file in uploaded_files:

        # Save temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
            temp_file.write(uploaded_file.read())
            temp_pdf_path = temp_file.name

        # Read PDF
        pdf_reader = PyPDF2.PdfReader(temp_pdf_path)

        # Read pages
        for page in pdf_reader.pages:
            text = page.extract_text()

            if text:
                raw_text += text


    # ---------------------------------------------------------
    # SPLIT TEXT INTO CHUNKS
    # ---------------------------------------------------------

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    texts = text_splitter.split_text(raw_text)

    st.success(f"Total Text Chunks Created: {len(texts)}")


    # ---------------------------------------------------------
    # CREATE EMBEDDINGS
    # ---------------------------------------------------------

    embeddings = HuggingFaceEmbeddings(
        model_name=embedding_model
    )


    # ---------------------------------------------------------
    # CREATE OR LOAD FAISS
    # ---------------------------------------------------------

    if os.path.exists(faiss_index_path):

        docsearch = FAISS.load_local(
            faiss_index_path,
            embeddings,
            allow_dangerous_deserialization=True
        )

        st.success("FAISS Index Loaded Successfully")

    else:

        docsearch = FAISS.from_texts(texts, embeddings)

        docsearch.save_local(faiss_index_path)

        st.success("New FAISS Index Created Successfully")


    # ---------------------------------------------------------
    # LOAD OLLAMA MODEL
    # ---------------------------------------------------------

    llm = OllamaLLM(
        model=ollama_model
    )


    # ---------------------------------------------------------
    # CREATE MEMORY
    # ---------------------------------------------------------

    memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True,
        output_key='answer'
    )


    # ---------------------------------------------------------
    # CREATE CONVERSATIONAL CHAIN
    # ---------------------------------------------------------

    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=docsearch.as_retriever(search_kwargs={"k": 3}),
        memory=memory,
        return_source_documents=True
    )


    # ---------------------------------------------------------
    # CHAT HISTORY STORAGE
    # ---------------------------------------------------------

    if "messages" not in st.session_state:
        st.session_state.messages = []


    # ---------------------------------------------------------
    # DISPLAY OLD MESSAGES
    # ---------------------------------------------------------

    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.markdown(message["content"])


    # ---------------------------------------------------------
    # USER QUESTION INPUT
    # ---------------------------------------------------------

    prompt = st.chat_input("Ask questions about your PDFs...")


    # ---------------------------------------------------------
    # GENERATE ANSWER
    # ---------------------------------------------------------

    if prompt:

        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Store user message
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })


        # Generate AI response
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                result = qa.invoke({"question": prompt})

                answer = result["answer"]

                st.markdown(answer)


                # ---------------------------------------------------------
                # SHOW SOURCE REFERENCES
                # ---------------------------------------------------------

                st.subheader("📚 Source References")

                source_docs = result["source_documents"]

                for i, doc in enumerate(source_docs):

                    st.markdown(f"### Source {i+1}")

                    st.write(doc.page_content[:500])

                    st.divider()


        # Store assistant message
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

else:

    st.info("Please upload PDF files to start chatting.")

