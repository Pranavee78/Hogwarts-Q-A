from store import *
from langchain_ollama import OllamaLLM
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from store import load_vectorstore
import numpy as np
from langchain_chroma import Chroma

ollama = Ollama(
    base_url='http://localhost:11434',
    model="llama3:latest",
    
)

def query(question, vectorstore, top_n = 10):
    retriever = vectorstore.as_retriever(search_kwargs={"k": top_n})
    docs = retriever.invoke(question)
    # print(len(docs))
    
    qachain = RetrievalQA.from_chain_type(ollama, retriever=retriever)
    res = qachain.invoke({"query": question})
    print(res['result'])
    
    
def query_2(question, vectorstore, top_n=10, send_prompt = True):
    # Convert the query to an embedding
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = embedding_model.encode([question])[0].tolist()
    
    # Retrieve the top N most relevant documents
    docs = vectorstore.similarity_search_by_vector(query_embedding, k=top_n)
    print(f"Retrieved {len(docs)} relevant documents")
    
    context = ""
    for doc in docs:
        text = doc.page_content
        metadata = doc.metadata if hasattr(doc, 'metadata') else {}  # Retrieve metadata if available
        chapter = metadata.get('chapter_name', 'Unknown Chapter')  # Example metadata field: 'chapter_name'
        print(chapter)
        
        context += f"Meta: {chapter}\n{text}\n\n"

    print(context)
    
    # Prepare the prompt for the language model
    if send_prompt:
        prompt = f"As a wizard steeped in the magical world of Harry Potter, respond to the following question only using the references provided. Make sure to include specific quotes and references from the text to explain your answer in a style that J.K. Rowling might use.\n\nQuestion: {question}\n\nContext: {context}\n\nAnswer:"
        # prompt = f"Based on the following context, answer the question: {question}\n\nContext: {context}\n\nAnswer:"
        
        # Use Ollama to generate the answer
        response = ollama.invoke(prompt)
        
        return response
    else:
        return context

# Main execution
if __name__ == "__main__":
    vectorstore = load_vectorstore("chroma_store_meta")
    print("Vector store loaded")
    question = "Who is Sirius Black to Harry Potter, and what role does he play in his life?"
    # query(question, vectorstore)
    print("query_2")   
    print(query_2(question,vectorstore))
    while(True):
        question = input("enter the question:")
        if question == '\q':
            break
        # query(question,vectorstore) 
        print("query_2")   
        print(query_2(question,vectorstore))
        