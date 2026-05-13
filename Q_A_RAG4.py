# ---------------------------------------------------------
# IMPORT REQUIRED LIBRARIES
# ---------------------------------------------------------

# Streamlit for web UI
import streamlit as st

# OS module for file handling
import os

# PyPDF2 for reading PDF files
import PyPDF2

# YAML file reader
import yaml

# Text splitter for chunking large text
from langchain_text_splitters import CharacterTextSplitter

# FAISS vector database
from langchain_community.vectorstores import FAISS

# HuggingFace embeddings
from langchain_community.embeddings import HuggingFaceEmbeddings

# Ollama local LLM
from langchain_ollama import OllamaLLM

# Retrieval QA chain
from langchain_classic.chains import RetrievalQA


# ---------------------------------------------------------
# LOAD YAML CONFIG FILE
# ---------------------------------------------------------

# Open config.yaml file
with open("config.yaml", "r") as file:

    # Load YAML data into Python dictionary
    config = yaml.safe_load(file)


# ---------------------------------------------------------
# READ VALUES FROM YAML FILE
# ---------------------------------------------------------

# PDF folder path
pdf_folder_path = config["pdf_folder_path"]

# FAISS save path
faiss_index_path = config["faiss_index_path"]

# Ollama model name
ollama_model = config["ollama_model"]

# Embedding model name
embedding_model = config["embedding_model"]

# Chunk settings
chunk_size = config["chunk_size"]
chunk_overlap = config["chunk_overlap"]


# ---------------------------------------------------------
# STREAMLIT PAGE TITLE
# ---------------------------------------------------------

st.title("Local RAG PDF Chatbot")


# ---------------------------------------------------------
# EXTRACT TEXT FROM PDF FILES
# ---------------------------------------------------------

# Variable to store extracted text
raw_text = ""

# Loop through all files in folder
for file in os.listdir(pdf_folder_path):

    # Process only PDF files
    if file.endswith(".pdf"):

        # Create full PDF path
        pdf_path = os.path.join(pdf_folder_path, file)

        # Read PDF
        pdf_reader = PyPDF2.PdfReader(pdf_path)

        # Loop through all pages
        for page in pdf_reader.pages:

            # Extract text from page
            text = page.extract_text()

            # Add extracted text
            if text:
                raw_text += text


# ---------------------------------------------------------
# SPLIT TEXT INTO CHUNKS
# ---------------------------------------------------------

# Create text splitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    length_function=len
)

# Split text into chunks
texts = text_splitter.split_text(raw_text)

# Display number of chunks
st.write("Number of text chunks:", len(texts))


# ---------------------------------------------------------
# CREATE HUGGINGFACE EMBEDDINGS
# ---------------------------------------------------------

# Load local embedding model
embeddings = HuggingFaceEmbeddings(
    model_name=embedding_model
)


# ---------------------------------------------------------
# CREATE OR LOAD FAISS VECTOR DATABASE
# ---------------------------------------------------------

# Check if FAISS index already exists
if os.path.exists(faiss_index_path):

    # Load existing FAISS index
    docsearch = FAISS.load_local(
        faiss_index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    st.success("FAISS index loaded successfully!")

else:

    # Create new FAISS index
    docsearch = FAISS.from_texts(texts, embeddings)

    # Save FAISS index locally
    docsearch.save_local(faiss_index_path)

    st.success("FAISS index created and saved successfully!")


# ---------------------------------------------------------
# LOAD OLLAMA MODEL
# ---------------------------------------------------------

# Load local Ollama model
llm = OllamaLLM(
    model=ollama_model
)


# ---------------------------------------------------------
# CREATE RETRIEVAL QA CHAIN
# ---------------------------------------------------------

# Create RAG pipeline
qa = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=docsearch.as_retriever()
)


# ---------------------------------------------------------
# USER INPUT
# ---------------------------------------------------------

# Text input box
query = st.text_input("Ask a question about your PDFs:")

# Generate button
generate_button = st.button("Generate Answer")


# ---------------------------------------------------------
# GENERATE ANSWER
# ---------------------------------------------------------

# Check if button clicked and query exists
if generate_button and query:

    # Loading spinner
    with st.spinner("Generating answer..."):

        # Generate answer
        response = qa.run(query)

        # Display answer
        st.subheader("Answer:")
        st.write(response)
