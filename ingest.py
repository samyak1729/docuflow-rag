from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
try:
    from pdf2image import convert_from_path
    import pytesseract
except ImportError:
    convert_from_path = None
    pytesseract = None

def load_doc(file_path):
    print(f"Processing: {file_path}")
    try:
        if file_path.endswith(".pdf"):
            try:
                docs = PyPDFLoader(file_path).load()
                if not docs or all(not doc.page_content.strip() for doc in docs):
                    raise ValueError("No text extracted - possibly scanned PDF")
                print(f"Loaded {file_path} as digital PDF: {len(docs)} pages")
                return docs
            except Exception as e:
                if convert_from_path and pytesseract:
                    print(f"{file_path} might be scanned: {e}. Trying OCR...")
                    images = convert_from_path(file_path)
                    text = "".join(pytesseract.image_to_string(img) for img in images)
                    if not text.strip():
                        print(f"OCR failed to extract text from {file_path}")
                        return None
                    print(f"OCR extracted {len(text)} chars from {file_path}")
                    return [{"page_content": text, "metadata": {"source": file_path}}]
                else:
                    print(f"OCR not available for {file_path}: {e}")
                    return None
        elif file_path.endswith(".docx"):
            docs = Docx2txtLoader(file_path).load()
            if not docs or all(not doc.page_content.strip() for doc in docs):
                print(f"Warning: {file_path} loaded but no text extracted")
                return None
            total_text = "\n".join(doc.page_content for doc in docs)
            print(f"Loaded {file_path} as Docx: {len(docs)} docs, {len(total_text)} chars")
            return docs
        elif file_path.endswith(".md"):
            docs = TextLoader(file_path).load()
            print(f"Loaded {file_path} as markdown: {len(docs)} docs")
            return docs
        else:
            print(f"Unsupported file type: {file_path}")
            return None
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def split_docs(docs, file_path, chunk_size=500, chunk_overlap=50):
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = splitter.split_documents(docs)
    print(f"Split {file_path} into {len(chunks)} chunks")
    return chunks

if __name__ == "__main__":
    folder = "sample_docs"
    os.makedirs(folder, exist_ok=True)
    all_chunks = []
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        docs = load_doc(file_path)
        if docs:
            chunks = split_docs(docs, file_path)
            all_chunks.extend(chunks)
    print(f"Total got {len(all_chunks)} chunks")
    # Print first 5 and last chunk
    for i, chunk in enumerate(all_chunks[:5]):
        print(f"Chunk {i}: {chunk.page_content[:100]}...")
    if all_chunks:
        print(f"Last Chunk ({len(all_chunks)-1}): {all_chunks[-1].page_content[:100]}...")
