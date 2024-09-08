# Opensource PDF ChatBot with Groq

This code implements a Retrieval-Augmented Generation (RAG) based conversational AI application using Streamlit. It allows users to upload and process PDF documents, then interact with the content through a chatbot interface. The chatbot leverages advanced Natural Language Processing (NLP) techniques to answer user queries based on the information contained within the uploaded PDFs.

# Table of Contents

## Features

1. Upload and process PDF documents
2. Chatbot interface for user interaction
3. Answer user queries based on PDF content

## Technologies Used

1. **Streamlit:** Web framework for building dashboards and data apps
2. **langchain:** Library for building NLP pipelines and creating conversational AI
3. **LlamaParse:** API for parsing and understanding documents
4. **Chroma:** Vector database for storing and querying document embeddings
5. **FastEmbedEmbeddings:** Model for generating document embeddings
6. **nltk:** Library for natural language processing tasks

## Project Structure:

1. main.py: Main script containing the Streamlit app logic
2. Helper functions for:
Loading/parsing uploaded PDF data using LlamaParse
Saving parsed documents to Markdown format
Creating and managing the Chroma vector database
Setting up the Q&A system using the retrieval model and vectorstore


## Additional Notes

1. The code utilizes caching mechanisms (@st.cache_data, @st.cache_resource) to improve performance.
2. A logging mechanism is implemented to record errors and track processing times.
3. Session state variables (st.session_state) are used to maintain the interaction history between the user and the chatbot.
