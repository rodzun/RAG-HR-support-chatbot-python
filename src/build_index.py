import os
import json
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from openai import OpenAI

load_dotenv()

# Config for OpenRouter
api_key = os.getenv("OPENROUTER_API_KEY")
embedding_model = os.getenv("EMBEDDING_MODEL", "openai/text-embedding-3-small")

# Set up embeddings with OpenRouter
embeddings = OpenAIEmbeddings(
    model=embedding_model,
    openai_api_key=api_key,
    openai_api_base="https://openrouter.ai/api/v1",
)

# Load document
loader = TextLoader("data/faq_document.txt")
documents = loader.load()

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=len,
)
chunks = text_splitter.split_documents(documents)

if len(chunks) < 20:
    raise ValueError("Document must generate at least 20 chunks.")

# Save chunks for inspection
chunks_list = [chunk.page_content for chunk in chunks]
with open("outputs/chunks.json", "w") as f:
    json.dump(chunks_list, f, indent=2)
print(f"Chunks saved to outputs/chunks.json for review.")

# Create vector store
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("faiss_index")

print(f"Index built with {len(chunks)} chunks.")