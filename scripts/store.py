from text_extraction import *
from data_chunking import *
import os
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer
from langchain_chroma import Chroma



# Embedding class to use with Chroma
class SentenceTransformersEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text: str) -> list[float]:
        embedding = self.model.encode([text], show_progress_bar=True)[0]
        return embedding.tolist()
    
    
# Initialize SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')



def get_embeddings(text, embedding_model):
    # Ensure chunked_texts is a list of strings
    chunked_texts = get_chunk(text)
    
    # Check if chunked_texts is indeed a list of strings
    if not isinstance(chunked_texts, list) or not all(isinstance(t, str) for t in chunked_texts):
        raise ValueError("get_chunk must return a list of strings")
    
   
    chunk_embeddings = embedding_model.encode(chunked_texts, show_progress_bar=True)
    return chunk_embeddings

 


def store(chunked_text, embedding_model, batch_size=5000, persist_directory="chroma_store"):
    """
    Store the chunked text in Chroma vectorstore in smaller batches to avoid exceeding the maximum batch size.
    The vectorstore is persisted to a directory on disk.
    
    Args:
        chunked_text (list): The text chunks to be stored.
        embedding_model (SentenceTransformer): The Sentence Transformer model to generate embeddings.
        batch_size (int, optional): The size of each batch of text to process. Defaults to 5000.
        persist_directory (str, optional): The directory where the vectorstore will be saved. Defaults to "chroma_store".
    
    Returns:
        vectorstore: The Chroma vectorstore instance.
    """

    if not chunked_text:
        print("No text chunks to store.")
        return None

    # Ensure the persist directory exists
    os.makedirs(persist_directory, exist_ok=True)

    embeddings = SentenceTransformersEmbeddings(embedding_model)
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

    # Split chunked_text into smaller batches
    for i in range(0, len(chunked_text), batch_size):
        batch = chunked_text[i:i + batch_size]
        if not batch:
            continue
        vectorstore.add_texts(texts=batch)
    
    return vectorstore



def store_metadata(chunked_text_with_metadata, embedding_model, batch_size=5000, persist_directory="chroma_store_meta"):
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    embeddings = SentenceTransformersEmbeddings(embedding_model)
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

    seen_texts = set()  # Track stored texts to prevent duplicates

    for i in range(0, len(chunked_text_with_metadata), batch_size):
        batch = chunked_text_with_metadata[i:i + batch_size]
        
        texts = [item['text'] for item in batch]
        metadatas = [item['metadata'] for item in batch]

        # Remove duplicates
        unique_texts = []
        unique_metadatas = []
        for text, metadata in zip(texts, metadatas):
            if text not in seen_texts:
                seen_texts.add(text)
                unique_texts.append(text)
                unique_metadatas.append(metadata)

        if unique_texts:
            vectorstore.add_texts(texts=unique_texts, metadatas=unique_metadatas)

    return vectorstore



def load_vectorstore(persist_directory="chroma_store_meta"):
    """
    Load the persisted Chroma vectorstore from disk using the specified embedding model.
    
    Args:
        persist_directory (str, optional): The directory where the vectorstore is saved. Defaults to "chroma_store".
        embedding_model (SentenceTransformer): The Sentence Transformer model to generate embeddings.
    
    Returns:
        vectorstore: The loaded Chroma vectorstore instance.
    """
    # Initialize the custom embeddings with the SentenceTransformer model
    embeddings = SentenceTransformersEmbeddings(embedding_model)
    
    # Load the Chroma vectorstore from the persisted directory
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
    
    return vectorstore




# Main execution
if __name__ == "__main__":
    pdf_path = "harry_potter.pdf"
    book_text = load_and_parse_pdf(pdf_path)
    chapters = split_into_chapters(book_text)
    chunked_text = book_chunks(chapters)
    store(chunked_text, embedding_model)
    chunked_meta = book_chunks_meta(chapters)
    store_metadata(chunked_meta, embedding_model)
    load_vectorstore("chroma_store_meta")
    # print(get_embeddings("What is the relationship between Harry Potter and Sirius Black in Prisoner of Azkaban?",embedding_model).tolist())