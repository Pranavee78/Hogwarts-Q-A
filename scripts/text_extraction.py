import PyPDF2
import re

 ## reads pdf file and return a text file

def load_and_parse_pdf(file_path):
    try:
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return None
   
    
#splitting the extracted text into chapters

def split_into_chapters(book_text):
    import re
    chapters = book_text.split('Chapter')
    chapters = ["Chapter " + chapter.strip() for chapter in chapters if chapter.strip()]
    
    # Assuming Chapter 1 onwards are actual story chapters
    if len(chapters) > 1:
        return chapters[1:]  # Skip the intro part
    return chapters

# to check if only the chapters are loaded

def print_chapter_previews(chapters):
    for i, ch in enumerate(chapters):
        preview = ch[:300].replace("\n", " ")  # Replace newlines for cleaner output
        print(f"Chapter {i+1} Preview: {preview}...\n{'-'*80}")


# Example usage
file_path = 'harry_potter.pdf'
book_text = load_and_parse_pdf(file_path)
if book_text:
    print("PDF loaded and parsed successfully.")
    chapters = split_into_chapters(book_text)
    print(f"Book split into {len(chapters)} chapters.")
    print_chapter_previews(chapters)


