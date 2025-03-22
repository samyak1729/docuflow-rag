from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os

def load_doc(file_path):
    try:
        if file_path.endswith(".md"):
            return TextLoader(file_path).load()
        return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def split_docs(docs, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

if __name__ == "__main__":
    folder = "sample_docs"
    os.makedirs(folder, exist_ok=True)
    all_chunks = []
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        docs = load_doc(file_path)
        if docs:
            chunks = split_docs(docs)
            all_chunks.extend(chunks)
    print(f"Got {len(all_chunks)} chunks")
    for i, chunk in enumerate(all_chunks[:5]):
        print(f"Chunk {i}: {chunk.page_content[:100]}...")
