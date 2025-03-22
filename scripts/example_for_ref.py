# Import required modules
from sentence_transformers import SentenceTransformer
from langchain.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.embeddings.base import Embeddings
from langchain.llms import Ollama
from langchain.document_loaders import WebBaseLoader

# Step 1: Load the document and split it into chunks
loader = WebBaseLoader("https://www.gutenberg.org/files/1727/1727-h/1727-h.htm")
data = loader.load()

# Split text into chunks for embedding
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# Step 2: Use all-MiniLM-L6-v2 from Sentence Transformers to generate embeddings
class SentenceTransformersEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.model.encode(texts, show_progress_bar=True)

    def embed_query(self, text: str) -> list[float]:
        return self.model.encode([text], show_progress_bar=True)[0]

# Initialize the embedding model
embedding_model = SentenceTransformersEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 3: Store the document embeddings using Chroma vector store
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embedding_model)

# Step 4: Use Llama 3.1 with Ollama for question answering
llm = Ollama(model="llama3:latest", base_url="http://localhost:11434")  # Ensure ollama serve is running

# Step 5: Build the RetrievalQA chain
retriever = vectorstore.as_retriever()
qachain = RetrievalQA.from_chain_type(llm, retriever=retriever)

# Step 6: Ask a question
question = "Who is Neleus and who is in Neleus' family?"
res = qachain.invoke({"query": question})

# Output the result
print(res['result'])