from text_extraction import load_and_parse_pdf

# Extract text from the PDF
data = load_and_parse_pdf("book.pdf")

# LangChain expects document objects with a `page_content` attribute
# Wrap the extracted text into a document object (as a list)
from langchain.docstore.document import Document

documents = [Document(page_content=data)]

# Use the text splitter on the document
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(documents)

# Embeddings and Vector Store setup
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import Chroma

oembed = OllamaEmbeddings(base_url="http://localhost:11434", model="llama3:latest")
vectorstore = Chroma.from_documents(documents=all_splits, embedding=oembed)

# Ask a question
question = "What is the relationship between Harry Potter and Sirius Black in Prisoner of Azkaban?"
docs = vectorstore.similarity_search(question)

# Print the number of relevant documents
print(len(docs))


def main():
    from langchain_community.llms import Ollama
    ollama = Ollama(
        base_url='http://localhost:11434',
        model="llama3:latest"
    )
    from langchain.chains import RetrievalQA
    qachain=RetrievalQA.from_chain_type(ollama, retriever=vectorstore.as_retriever())
    res = qachain.invoke({"query": question})
    print(res['result'])