import os
import textwrap
import joblib
import logging
import hashlib
import nest_asyncio
import tempfile
import time
import shutil
import streamlit as st
from pathlib import Path
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
from langchain_groq import ChatGroq
from llama_parse import LlamaParse
from langchain_community.vectorstores import Chroma

import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

# Set up logging
logging.basicConfig(filename='app.log', filemode='w',
                    format='%(name)s - %(levelname)s - %(message)s',
                    level=logging.DEBUG)

# Initialize APIs from Streamlit secrets
groq_api_key = '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'
llamaparse_api_key = 'llx-vnhbLwp1frdxccOj1ZqdGiSvP8mJQo4FQ4gQjYS1Fj4xkqUM'

# Apply nest_asyncio to handle async issues
nest_asyncio.apply()

@st.cache_data
def load_or_parse_data(uploaded_file, directory_name):
    """Load or parse PDF data."""
    if not hasattr(uploaded_file, 'read'):
        st.error("Invalid file uploaded.")
        return None

    temp_file_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            temp_file.write(uploaded_file.read())
            temp_file_path = temp_file.name

        data_file = os.path.join(directory_name, f"parsed_data.pkl")

        if os.path.exists(data_file):
            logging.info(f"Loading parsed data from file: {data_file}.")
            parsed_data = joblib.load(data_file)
        else:
            parsing_instruction = """
            The provided document is a leave policy for all the employees filed by Cognizant,
            with all the days that an employee can avail. This form provides detailed information
            about the company's leave policy that can be availed by an employee.
            It includes policy statements, and other relevant disclosures required by the employee.
            It contains many tables. Try to be precise while answering the questions.
            """
            parser = LlamaParse(api_key=llamaparse_api_key, result_type="markdown", parsing_instruction=parsing_instruction)

            parsed_data = parser.load_data(temp_file_path)
            os.makedirs(directory_name, exist_ok=True)
            joblib.dump(parsed_data, data_file)
            logging.info(f"Parsed data saved to file: {data_file}.")

        return parsed_data

    except Exception as e:
        logging.error(f"Error loading or parsing data: {e}")
        st.error(f"Failed to process file: {e}")
        return None
    finally:
        if temp_file_path:
            os.remove(temp_file_path)

def save_to_markdown(llama_parse_documents, directory_name):
    """Save parsed documents to markdown file."""
    markdown_path = os.path.join(directory_name, "output.md")
    try:
        with open(markdown_path, 'w', encoding='utf-8') as f:
            for doc in llama_parse_documents:
                f.write(doc.text + '\n')
        return markdown_path
    except Exception as e:
        logging.error(f"Error in save_to_markdown: {e}")
        st.error(f"Error occurred while saving to markdown: {e}")
        return None

@st.cache_resource
def create_vector_database(markdown_path, directory_name):
    """Create or load vector database from markdown file."""
    try:
        loader = UnstructuredMarkdownLoader(markdown_path)
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)
        docs = text_splitter.split_documents(documents)

        embed_model = FastEmbedEmbeddings(model_name="BAAI/bge-base-en-v1.5")

        vs = Chroma.from_documents(
            documents=docs,
            embedding=embed_model,
            persist_directory=directory_name,
            collection_name="rag"
        )

        logging.info('Vector DB created successfully!')
        return vs, embed_model

    except Exception as e:
        logging.error(f"Error in create_vector_database: {e}")
        st.error(f"Error occurred while creating the vector database: {e}")
        return None, None

@st.cache_resource
def setup_qa_system(_embed_model, directory_name):
    """Setup the Q&A system using the Chroma vectorstore."""
    try:
        chat_model = ChatGroq(temperature=0, model_name="mixtral-8x7b-32768", api_key=groq_api_key)
        vectorstore = Chroma(embedding_function=_embed_model, persist_directory=directory_name, collection_name="rag")
        retriever = vectorstore.as_retriever(search_kwargs={'k': 2})

        prompt = PromptTemplate(
            template="""
            Use the following pieces of information to answer the user's question.
            If you don't know the answer, just say that you don't know, don't try to make up an answer.

            Context: {context}
            Question: {question}

            Only return the helpful answer below and nothing else.
            Helpful answer:
            """,
            input_variables=['context', 'question']
        )

        qa = RetrievalQA.from_chain_type(llm=chat_model, chain_type="stuff", retriever=retriever, return_source_documents=True, chain_type_kwargs={"prompt": prompt, "verbose": True})

        logging.info("Q&A system setup successfully.")
        return qa

    except Exception as e:
        logging.error(f"Error in setup_qa_system: {e}")
        st.error(f"Error occurred while setting up the Q&A system: {e}")
        return None

def main():
    """Main function to run the Streamlit app."""
    # Initialize session state variables
    if 'interaction_history' not in st.session_state:
        st.session_state.interaction_history = []

    if 'qa_chain' not in st.session_state:
        st.session_state.qa_chain = None

    st.sidebar.title("File Upload")
    uploaded_file = st.sidebar.file_uploader("Upload a PDF file", type="pdf")

    if uploaded_file is not None:
        # Create a unique directory for each uploaded file
        directory_name = f"session_{hashlib.md5(uploaded_file.name.encode()).hexdigest()}"

        with st.spinner('Processing PDF...'):
            parsed_data = load_or_parse_data(uploaded_file, directory_name)
            if parsed_data:
                markdown_path = save_to_markdown(parsed_data, directory_name)
                if markdown_path:
                    vs, embed_model = create_vector_database(markdown_path, directory_name)
                    if vs and embed_model:
                        st.session_state.qa_chain = setup_qa_system(embed_model, directory_name)

    if st.session_state.qa_chain:
        st.title("Chat with PDF")
        user_input = st.chat_input("You:")
        if user_input:
            start_time = time.time()
            st.session_state.interaction_history.append({"role": "user", "message": user_input})
            st.chat_message("user").markdown(user_input)
            with st.chat_message("assistant"):
                with st.spinner("Assistant is typing..."):
                    response = st.session_state.qa_chain.invoke({"query": user_input})
                    full_response = response['result']
                    st.markdown(full_response)

            st.session_state.interaction_history.append({"role": "assistant", "message": full_response})
            logging.warning(f"Query: {user_input} - Answer: {full_response} - Processing Time: {round(time.time() - start_time, 2)} seconds")

if __name__ == "__main__":
    main()
