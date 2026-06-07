Project Highlights
Built an end-to-end RAG pipeline
Implemented semantic document retrieval
Integrated Google Gemini for context-aware responses
Designed a scalable architecture for document question answering

PDF RAG Chatbot
A Retrieval-Augmented Generation (RAG) chatbot that answers questions from PDF documents using LangChain, FAISS, Hugging Face Embeddings, and Google Gemini.

- Features:
Upload and process PDF documents
Automatic document chunking and preprocessing
Semantic search using FAISS vector database
Context-aware question answering with Google Gemini
Retrieval-Augmented Generation (RAG) pipeline
Fast and accurate document-based responses

-Tech Stack:
Python
LangChain
FAISS Vector Store
Hugging Face Sentence Transformers
Google Gemini API
Streamlit
PyPDF

-Project Workflow:
Load PDF documents
Split text into chunks
Generate embeddings using Sentence Transformers
Store embeddings in FAISS
Retrieve relevant chunks based on user queries
Generate answers using Google Gemini and retrieved context

-Architecture:

PDF Document
→ Text Chunking
→ Embedding Generation
→ FAISS Vector Database
→ Semantic Retrieval
→ Gemini LLM
→ Final Answer
