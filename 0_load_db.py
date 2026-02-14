import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from smolagents import CodeAgent, HfApiModel, DuckDuckGoSearchTool, Tool
import gradio as gr
import os
import glob
from typing import List
import re


PROJECT_NAME = "my_project"


# ==================== 1. SETUP VECTOR DB ====================
# Persistent storage in ./chroma_db folder
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# Local embeddings (free, offline, fast for internal use)
# all-MiniLM-L6-v2 has a 256 token limit, but optimal chunking is typically 128-256 tokens
# Using ~200 tokens as a safe default for good semantic coherence
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OPTIMAL_TOKENS_PER_CHUNK = 200  # Conservative default for this model
TOKENS_PER_WORD = 1.3  # Approximate ratio for English text

embedding_fn = SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME  # 384-dim, good balance of speed/quality
)

collection = chroma_client.get_or_create_collection(
    name=f"{PROJECT_NAME}_knowledge", 
    embedding_function=embedding_fn,
    metadata={"hnsw:space": "cosine"}
)

# ==================== 2. DOCUMENT INGESTION PIPELINE ====================

def estimate_tokens(text: str) -> int:
    """Rough estimation of token count based on word count."""
    word_count = len(text.split())
    return int(word_count * TOKENS_PER_WORD)

def chunk_text(text: str, target_tokens: int = OPTIMAL_TOKENS_PER_CHUNK) -> List[str]:
    """
    Split text into chunks of approximately target_tokens.
    Uses sentence-aware splitting to preserve semantic coherence.
    """
    words = text.split()
    target_words = int(target_tokens / TOKENS_PER_WORD)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for word in words:
        current_chunk.append(word)
        current_word_count += 1
        
        # Check if we've reached target size and are at a sentence boundary
        if current_word_count >= target_words and word.endswith(('.', '!', '?')):
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_word_count = 0
    
    # Add any remaining words
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def read_file(file_path: str) -> str:
    """Read file contents with encoding fallback."""
    encodings = ['utf-8', 'latin-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Could not decode file: {file_path}")

def ingest_documents(
    documents_dir: str = "./documents",
    target_tokens: int = OPTIMAL_TOKENS_PER_CHUNK
) -> dict:
    """
    Traverse documents directory, chunk all files, and add to vector DB.
    
    Returns:
        dict: Statistics about ingestion process
    """
    if not os.path.exists(documents_dir):
        print(f"Documents directory '{documents_dir}' does not exist. Skipping ingestion.")
        return {"status": "skipped", "reason": "directory_not_found"}
    
    # Supported text file extensions
    text_extensions = {'.txt', '.md', '.py', '.js', '.ts', '.jsx', '.tsx', 
                       '.json', '.yaml', '.yml', '.csv', '.log', '.rst'}
    
    stats = {
        "files_processed": 0,
        "chunks_created": 0,
        "total_documents_before": collection.count(),
        "errors": []
    }
    
    # Find all files recursively
    all_files = []
    for ext in text_extensions:
        pattern = os.path.join(documents_dir, f"**/*{ext}")
        all_files.extend(glob.glob(pattern, recursive=True))
    
    print(f"Found {len(all_files)} files to process in '{documents_dir}'")
    
    for file_path in all_files:
        try:
            # Read file content
            content = read_file(file_path)
            
            if not content.strip():
                continue
            
            # Get relative path for metadata
            rel_path = os.path.relpath(file_path, documents_dir)
            
            # Split into chunks
            chunks = chunk_text(content, target_tokens)
            
            if not chunks:
                continue
            
            # Prepare metadata for each chunk
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                metadata = {
                    "source": rel_path,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "file_type": os.path.splitext(file_path)[1],
                    "estimated_tokens": estimate_tokens(chunk)
                }
                metadatas.append(metadata)
                # Create unique ID: filename_chunk_index
                safe_id = re.sub(r'[^a-zA-Z0-9_-]', '_', f"{rel_path}_{i}")
                ids.append(safe_id)
            
            # Add to collection
            collection.add(
                documents=chunks,
                ids=ids,
                metadatas=metadatas
            )
            
            stats["files_processed"] += 1
            stats["chunks_created"] += len(chunks)
            print(f"  ✓ {rel_path}: {len(chunks)} chunks")
            
        except Exception as e:
            error_msg = f"Error processing {file_path}: {str(e)}"
            stats["errors"].append(error_msg)
            print(f"  ✗ {error_msg}")
    
    stats["total_documents_after"] = collection.count()
    stats["documents_added"] = stats["total_documents_after"] - stats["total_documents_before"]
    
    print(f"\nIngestion complete:")
    print(f"  Files processed: {stats['files_processed']}")
    print(f"  Chunks created: {stats['chunks_created']}")
    print(f"  Total documents in DB: {stats['total_documents_after']}")
    if stats["errors"]:
        print(f"  Errors: {len(stats['errors'])}")
    
    return stats

def clear_collection():
    """Clear all documents from the collection."""
    try:
        chroma_client.delete_collection("project_knowledge")
        global collection
        collection = chroma_client.get_or_create_collection(
            name="project_knowledge", 
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        print("Collection cleared successfully.")
    except Exception as e:
        print(f"Error clearing collection: {e}")

# Seed with sample data (run once, then comment out)
def init_knowledge():
    docs = [
        "Project Phoenix uses React frontend with FastAPI backend",
        "Authentication is handled via SSO with Azure AD",
        "Database schema: Users table has id, email, role, created_at",
        "Deployment pipeline: GitHub Actions → Docker → AWS ECS",
        "API rate limits: 1000 requests/hour per API key"
    ]
    collection.add(
        documents=docs,
        ids=[f"doc_{i}" for i in range(len(docs))],
        metadatas=[{"source": "internal_wiki"} for _ in docs]
    )

# ==================== 3. RUN INGESTION (Optional) ====================
if __name__ == "__main__":
    # Uncomment to run ingestion on startup:
    # ingest_documents()
    
    # Or with custom token target:
    # ingest_documents(target_tokens=150)  # Smaller chunks for more granular retrieval
    
    pass
