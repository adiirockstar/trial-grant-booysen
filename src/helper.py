from pathlib import Path
import PyPDF2
from docx import Document as DocxDocument

def load_documents(data_dir):
    """Load documents from a directory - supports txt, md, pdf, docx"""
    docs = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Warning: Data directory {data_dir} does not exist")
        return []
    
    # Text files
    for file_path in data_path.glob("**/*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                doc = type('Document', (), {'page_content': content, 'metadata': {'source': str(file_path)}})()
                docs.append(doc)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Markdown files
    for file_path in data_path.glob("**/*.md"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                doc = type('Document', (), {'page_content': content, 'metadata': {'source': str(file_path)}})()
                docs.append(doc)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    # PDF files
    for file_path in data_path.glob("**/*.pdf"):
        try:
            with open(file_path, 'rb') as f:
                pdf_reader = PyPDF2.PdfReader(f)
                content = ""
                for page in pdf_reader.pages:
                    content += page.extract_text() + "\n"
                doc = type('Document', (), {'page_content': content, 'metadata': {'source': str(file_path)}})()
                docs.append(doc)
        except Exception as e:
            print(f"Error loading PDF {file_path}: {e}")

    # DOCX files
    for file_path in data_path.glob("**/*.docx"):
        try:
            docx_doc = DocxDocument(file_path)
            content = ""
            for paragraph in docx_doc.paragraphs:
                content += paragraph.text + "\n"
            doc = type('Document', (), {'page_content': content, 'metadata': {'source': str(file_path)}})()
            docs.append(doc)
        except Exception as e:
            print(f"Error loading DOCX {file_path}: {e}")
    
    return docs