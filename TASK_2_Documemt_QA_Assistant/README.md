Document Q&A Assistant using RAG and Vector Embeddings

Project Overview

This project is an AI-powered Document Question Answering Assistant built using Retrieval-Augmented Generation (RAG).

It allows users to upload PDF documents and ask questions based on their content. The application retrieves relevant information from the uploaded document and generates accurate, context-based answers using Google's Gemini AI.

Features

- PDF document upload
- Automatic document processing
- Text extraction and chunking
- Vector embeddings
- Semantic search and retrieval
- AI-powered question answering
- Source page references
- Readable Markdown-formatted answers
- Streamlit web interface

Technologies Used

- Python
- Streamlit
- LangChain
- Google Gemini API
- ChromaDB
- Vector Embeddings

Project Structure

datheon_task/
│
├── main.py
├── requirements.txt
├── .gitignore
├── README.md
└── .env (not uploaded)

Installation and Setup

1. Clone the repository:

git clone YOUR_GITHUB_REPOSITORY_URL

2. Navigate to the project folder:

cd datheon_task

3. Create and activate a virtual environment.

4. Install dependencies:

pip install -r requirements.txt

5. Create a ".env" file and add your Gemini API key:

GOOGLE_API_KEY=your_api_key_here

6. Run the application:

streamlit run main.py

How to Use

1. Open the application in your browser.
2. Upload a PDF document.
3. Click Process Document.
4. Enter a question about the document.
5. Click Get Answer.
6. View the generated answer and source pages.

AI Model

Google Gemini 3.5 Flash

Project Type

AI / Machine Learning / Generative AI / Retrieval-Augmented Generation (RAG)

Author

Vaishnavi Gupta