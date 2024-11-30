import os
import torch
from typing import List, Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document

def load_documents(directory: str, glob_pattern: str = "**/*.pdf") -> List[Document]:
    """
    Load documents from a specified directory with a glob pattern.
    
    Args:
        directory (str): Path to the directory containing documents
        glob_pattern (str, optional): Pattern to match files. Defaults to PDF files.
    
    Returns:
        List[Document]: List of loaded documents
    """
    loader = DirectoryLoader(
        directory, 
        glob=glob_pattern,
        loader_cls=PyPDFLoader,  # More robust PDF loading
        show_progress=True
    )
    documents = loader.load()
    print(f"Number of documents loaded: {len(documents)}")
    return documents

def create_contextual_text_splitter(
    chunk_size: int = 500, 
    chunk_overlap: int = 50, 
    language: str = 'arabic'
) -> RecursiveCharacterTextSplitter:
    """
    Create a context-aware text splitter with language-specific separators.
    
    Args:
        chunk_size (int): Maximum size of text chunks
        chunk_overlap (int): Number of characters to overlap between chunks
        language (str): Language-specific text splitting configuration
    
    Returns:
        RecursiveCharacterTextSplitter: Configured text splitter
    """
    language_separators = {
        'arabic': ["\n\n", "\n", ".", "!", "?", "،", ";", ",", " ", ""],
        'english': ["\n\n", "\n", ".", "!", "?", ";", ",", " ", ""],
        'default': ["\n\n", "\n", ".", " ", ""]
    }
    
    separators = language_separators.get(language.lower(), language_separators['default'])
    
    return RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
        separators=separators
    )

def create_embeddings(
    model_name: str = "intfloat/multilingual-e5-large",
    device: Optional[str] = None
) -> HuggingFaceEmbeddings:
    """
    Create embeddings using a specified model.
    
    Args:
        model_name (str): Hugging Face model for embeddings
        device (str, optional): Compute device. Defaults to CUDA if available.
    
    Returns:
        HuggingFaceEmbeddings: Configured embedding model
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': device},
        encode_kwargs={'normalize_embeddings': True}
    )

def process_documents(
    directory: str, 
    persist_directory: str = "./db-KB",
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    language: str = 'arabic'
) -> Chroma:
    """
    Complete document processing pipeline.
    
    Args:
        directory (str): Source directory for documents
        persist_directory (str): Directory to store vector store
        chunk_size (int): Text chunk size
        chunk_overlap (int): Chunk overlap size
        language (str): Language for text splitting
    
    Returns:
        Chroma: Populated vector store
    """
    # Ensure persist directory exists
    os.makedirs(persist_directory, exist_ok=True)
    
    # Load documents
    documents = load_documents(directory)
    
    # Create contextual text splitter
    text_splitter = create_contextual_text_splitter(
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap, 
        language=language
    )
    
    # Split documents with context preservation
    texts = text_splitter.split_documents(documents)
    
    # Create vector store
    vectorstore = Chroma.from_documents(
        documents=texts, 
        embedding=create_embeddings(),
        persist_directory=persist_directory
    )
    
    print(f"Created vector store with {len(texts)} chunks in {persist_directory}")
    return vectorstore

def main():
    # Example usage
    vectorstore = process_documents(
        directory="Data", 
        persist_directory="./db-KB",
        chunk_size=500,
        chunk_overlap=50,
        language='arabic'
    )

if __name__ == "__main__":
    main()
