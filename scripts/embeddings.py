from text_extraction import *
from data_chunking import *
import os
from langchain_community.vectorstores import Chroma
from langchain.embeddings.base import Embeddings
from sentence_transformers import SentenceTransformer


# Embedding class to use with Chroma
class SentenceTransformersEmbeddings(Embeddings):
    def __init__(self, model):
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        try:
            embeddings = self.model.encode(texts, show_progress_bar=True)
            return [embedding.tolist() for embedding in embeddings]
        except Exception as e:
            print(f"Error generating embeddings: {e}")
            return []

    def embed_query(self, text: str) -> list[float]:
        try:
            embedding = self.model.encode([text], show_progress_bar=True)[0]
            return embedding.tolist()
        except Exception as e:
            print(f"Error generating query embedding: {e}")
            return []


# Initialize SentenceTransformer model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')


def get_embeddings(text, embedding_model):
    """ Generate embeddings for a given text using the embedding model. """
    chunked_texts = get_chunk(text)

    if not isinstance(chunked_texts, list) or not all(isinstance(t, str) for t in chunked_texts):
        raise ValueError("get_chunk must return a list of strings")

    try:
        chunk_embeddings = embedding_model.encode(chunked_texts, show_progress_bar=True)
        return chunk_embeddings
    except Exception as e:
        print(f"Error generating embeddings: {e}")
        return []


def store(chunked_text, embedding_model, batch_size=5000, persist_directory="chroma_store"):
    """ Store chunked text in Chroma vector database. """
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    embeddings = SentenceTransformersEmbeddings(embedding_model)
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

    for i in range(0, len(chunked_text), batch_size):
        batch = chunked_text[i:i + batch_size]
        if not batch:
            continue
        vectorstore.add_texts(texts=batch)

    vectorstore.persist()
    return vectorstore


def store_metadata(chunked_text_with_metadata, embedding_model, batch_size=5000, persist_directory="chroma_store_meta", verbose=False):
    """ Store chunked text with metadata in Chroma vector database. """
    if not os.path.exists(persist_directory):
        os.makedirs(persist_directory)

    embeddings = SentenceTransformersEmbeddings(embedding_model)
    vectorstore = Chroma(embedding_function=embeddings, persist_directory=persist_directory)

    for i in range(0, len(chunked_text_with_metadata), batch_size):
        batch = chunked_text_with_metadata[i:i + batch_size]
        if not batch:
            continue

        texts = [item['text'] for item in batch]
        metadatas = [item['metadata'] for item in batch]

        if verbose:
            print("\nBatch Metadata Preview:")
            for metadata, text in zip(metadatas, texts):
                print(f"Metadata: {metadata}, Text (first 100 chars): {text[:100]}")

        vectorstore.add_texts(texts=texts, metadatas=metadatas)
        vectorstore.persist()

    return vectorstore


def load_vectorstore(persist_directory="chroma_store_meta"):
    """ Load the persisted Chroma vectorstore from disk. """
    embeddings = SentenceTransformersEmbeddings(embedding_model)
    return Chroma(embedding_function=embeddings, persist_directory=persist_directory)


# Main execution
if __name__ == "__main__":
    pdf_path = "harry_potter.pdf"
    book_text = load_and_parse_pdf(pdf_path)
    chapters = split_into_chapters(book_text)
    # chunked_text = book_chunks(chapters)
    # store(chunked_text, embedding_model)
    chunked_meta = book_chunks_meta(chapters)
    store_metadata(chunked_meta, embedding_model, verbose=True)
    load_vectorstore("chroma_store_meta")

    # Example query embedding
    query_embedding = get_embeddings("What is the relationship between Harry Potter and Sirius Black in Prisoner of Azkaban?", embedding_model)

    print(f"Generated query embedding of size: {len(query_embedding[0])}")