
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_groq import ChatGroq
from langchain_core.documents import Document
from dotenv import load_dotenv
import pymupdf
import os

load_dotenv()

embedding_function = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.0,
    max_retries=2,
    groq_api_key=os.getenv("GROQ_API_KEY"),
    verbose=False,
)
retriever = None

def truncate_text(text, max_tokens=2048):
    return text[:max_tokens * 4]



def load_pdf_and_initialize_retriever(pdf_path: str):
    global retriever
    print(f"Loading PDF: {pdf_path}")
    doc = pymupdf.open(pdf_path)
    
    # Convert each page to a proper LangChain Document object
    docs = [Document(page_content=page.get_text()) for page in doc]
    
    # Split the documents into chunks
    splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    doc_chunks = splitter.split_documents(docs)
    
    # Embed and store
    vStore = Chroma.from_documents(doc_chunks, embedding_function, persist_directory="./vector_store")
    retriever = vStore.as_retriever()
    print("Retriever initialized.")

def get_bot_response(question: str) -> str:
    global retriever
    if retriever is None:
        return "PDF not loaded. Please upload a document first."

    docs = retriever.get_relevant_documents(question)
    combined_context = " ".join([doc.page_content for doc in docs])
    truncated_context = truncate_text(combined_context)
    prompt = f"Question: {question}\n\nContext: {truncated_context}\n\nAnswer:"
    response = llm.invoke(prompt)
    return response.content if response else "No response from the model."
