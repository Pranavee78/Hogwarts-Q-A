from text_extraction import *

from langchain.text_splitter import RecursiveCharacterTextSplitter



def book_chunks(chapters, chunk_size = 500, chunk_overlap = 100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,    
        chunk_overlap=chunk_overlap,   
    )
    chapter_chunks = []
    chunked_texts = []
    for chapter in chapters:
        chapter_chunks  = text_splitter.split_text(chapter)
        chunked_texts.extend(chapter_chunks) 
        
    print(f"Total Chunks: {len(chunked_texts)}")
    return chunked_texts

def get_chunk(text, chunk_size = 500, chunk_overlap = 100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,    
        chunk_overlap=chunk_overlap   
    )
    chunked_texts = []
    chunked_texts.extend(text_splitter.split_text(text))
    return chunked_texts

def book_chunks_meta(chapters,  chapter_names = [],chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunked_texts = []
    
    for i, chapter in enumerate(chapters):
        chapter_name = chapter_names[i] if i < len(chapter_names) else f"Chapter {i + 1}"
        chapter_chunks = text_splitter.split_text(chapter)
        
        # Add metadata to each chunk
        for chunk in chapter_chunks:
            chunked_texts.append({
                'text': chunk,
                'metadata': {
                    'chapter_name': chapter_name
                }
            })
    
    print(f"Total Chunks: {len(chunked_texts)}")
    return chunked_texts


if __name__ == "__main__":
    pdf_path = "harry_potter.pdf"
    book_text = load_and_parse_pdf(pdf_path)
    chapters = split_into_chapters(book_text)
    print(book_chunks_meta(chapters, []))