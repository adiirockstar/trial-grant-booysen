from pathlib import Path

def load_documents(data_dir):
    """Load documents from a directory"""
    docs = []
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"Warning: Data directory {data_dir} does not exist")
        return []
    
    # Simple document loader for text files
    for file_path in data_path.glob("**/*.txt"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # Create a simple document object
                doc = type('Document', (), {'page_content': content, 'metadata': {'source': str(file_path)}})()
                docs.append(doc)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    # Also handle markdown files
    for file_path in data_path.glob("**/*.md"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                doc = type('Document', (), {'page_content': content, 'metadata': {'source': str(file_path)}})()
                docs.append(doc)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
    
    return docs