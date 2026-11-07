import streamlit as st
import os
import tempfile

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
 

load_dotenv()

st.set_page_config(page_title="Document Q&A Assistant", page_icon="📚")

st.title("📚 Document Q&A Assistant")
st.write("Upload a PDF and ask questions about its content.")

api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    st.error("Google API key not found. Please check your .env file.")
    st.stop()

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(uploaded_file.getvalue())
        pdf_path = temp_file.name

    if st.button("Process Document"):
        with st.spinner("Reading and processing PDF..."):
            try:
                loader = PyPDFLoader(pdf_path)
                documents = loader.load()

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200
                )

                chunks = splitter.split_documents(documents)

                embeddings = GoogleGenerativeAIEmbeddings(
                    model="models/gemini-embedding-001",
                    google_api_key=api_key
                )

                vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                    collection_name="document_qa"
                )

                st.session_state.vectorstore = vectorstore
                st.success(f"Document processed! {len(chunks)} chunks created.")

            except Exception as e:
                st.error(f"Error: {e}")

if "vectorstore" in st.session_state:
    question = st.text_input("Ask a question about your PDF:")

    if st.button("Get Answer"):
        if question:
            with st.spinner("Finding your answer..."):
                try:
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-3.5-flash",
                        google_api_key=api_key,
                        temperature=0
                    )

                    docs = st.session_state.vectorstore.similarity_search(
                        question, k=3
                    )

                    context = "\n\n".join(
                        doc.page_content for doc in docs
                    )

                    prompt = f"""
                    Answer the question using only the context below.
                    If the answer is not present, say you cannot find it
                    in the uploaded document.

                    Context:
                    {context}

                    Question:
                    {question}
                    """

                    response = llm.invoke(prompt)
                    st.subheader("Answer")
                    if isinstance(response.content, str):
                       answer_text = response.content
                    else:
                     answer_text = "\n\n".join(
                      block.get("text", "")
                      for block in response.content
                      if isinstance(block, dict) and block.get("type") ==
                    "text"
                      )

                    st.markdown(answer_text)

                    st.subheader("Source Pages")
                    for doc in docs:
                        st.write(f"Page {doc.metadata.get('page', 0) + 1}")

                except Exception as e:
                    st.error(f"Error: {e}")