# base-agent-template (Your Robot Factory) 🤖

A "Just add water" template to build a lightweifght, but powerful custom agent for a given project.

#  Intelligent Document Agent

An AI-powered document ingestion and query system built with **ChromaDB**, **smolagents**, and **Gradio**. This project enables you to ingest documents from various formats (PDF, Word, Excel, images, code files, etc.), store them in a vector database, and query them through an intelligent agent interface.

---

## ✨ Features

- **📄 Multi-Format Document Support**: Extract text from numerous file formats including:
  - **Documents**: PDF, DOCX, TXT, MD, RST
  - **Spreadsheets**: XLSX, XLS, CSV
  - **Presentations**: PPTX, PPT
  - **Images**: PNG, JPG, JPEG, TIFF, BMP, GIF (with OCR)
  - **Code Files**: Python, JavaScript, TypeScript, JSX, TSX, JSON, YAML
  
- **🧠 Smart Chunking**: Text splitting optimized for `all-MiniLM-L6-v2` embeddings (~150-200 tokens per chunk)

- **🔍 Semantic Search**: Local vector database with cosine similarity search

- **🤖 AI Agent**: Powered by smolagents with tool-calling capabilities:
  - Internal document search
  - Web search (DuckDuckGo)
  - Wikipedia lookup
  - Webpage visiting
  - Python code execution
  - User input / human in the loop handling / asking for user feedback

- **🌐 Web Interface**: Gradio-based UI for easy interaction

- **⚡ CI/CD Ready**: GitHub Actions workflow for automated testing and deployment

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- [Hugging Face Token](https://huggingface.co/settings/tokens) (for model access via Fireworks AI)
- (Optional) Tesseract OCR for image text extraction

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/david-thrower/base-agent-template.git
   cd base-agent-template
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set environment variables**
   ```bash
   export HF_TOKEN="your_huggingface_token"
   export EMAIL_ID="your_email@example.com"
   ```

4. **Replace the files in the subdirectory  /documents with the documents you want your agent to know about.**
   `./documents`

5. **Run the ingestion pipeline (run only one time ever for each agent you create)**
   ```bash
   python 0_load_db.py
   ```

6. **Launch the agent**
   ```bash
   python 2_run_agent.py
   ```

---

## 📁 Project Structure

| File | Description |
|------|-------------|
| `0_load_db.py` | Document ingestion pipeline - extracts, chunks, and indexes documents into ChromaDB |
| `1_lint_agent.py` | CLI version of the agent for testing/linting (runs single query) |
| `2_run_agent.py` | Production agent with Gradio web interface |
| `requirements.txt` | Python dependencies |
| `.github/workflows/python-app.yml` | CI/CD pipeline configuration |

---

## 🔧 Configuration

### Document Ingestion (`0_load_db.py`)

Key settings you can modify:

```python
TARGET_TOKENS = 150  # Tokens per chunk (recommended: 128-256)
PROJECT_NAME = "my_project"  # Collection name prefix
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim embeddings
```

### Agent Configuration (`1_lint_agent.py` & `2_run_agent.py`)

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face API token for Fireworks AI inference |
| `EMAIL_ID` | Email for Wikipedia API user agent |
| `PROJECT_NAME` | Must match the name used in `0_load_db.py` |

---

## 🛠️ Tools Available to the Agent

| Tool | Purpose |
|------|---------|
| `internal_search` | Query your local document knowledge base |
| `web_search` | Search the web via DuckDuckGo |
| `visit_page` | Fetch and parse webpage content |
| `wikipedia` | Search Wikipedia articles |
| `python_interpreter` | Execute Python code |
| `user_input` | Request clarification from user |
| `final_answer` | Submit final response |

---

## 🔄 CI/CD Pipeline

The GitHub Actions workflow (`.github/workflows/python-app.yml`) automatically:

1. **Installs dependencies** from `requirements.txt`
2. **Lints code** with `flake8`
3. **Loads vector DB** by running `0_load_db.py`
4. **Tests agent** with `1_lint_agent.py`
5. **Deploys agent** by starting `2_run_agent.py` (background process)

**Required Secrets:**
- `HF_TOKEN` (GitHub Secret): Hugging Face API token
- `EMAIL_ID` (GitHub Variable): Contact email for Wikipedia API

---

## 📝 Usage Examples

### Ingest Documents
```python
# Ingest all supported files in ./documents
python 0_load_db.py

# The script will:
# - Recursively scan ./documents
# - Extract text from each file
# - Chunk into ~150 token segments
# - Store in ChromaDB with metadata
```

### Query via CLI (Lint Mode)
```python
# 1_lint_agent.py runs a single test query:
agent("Tell me all about the Platypus and similar mammals.")
```

### Interactive Web Interface
```bash
# 2_run_agent.py launches Gradio UI with shareable link
python 2_run_agent.py
# Output: Running on https://xxxx.gradio.live
```

---

## 🧩 Extending the System

### Adding New Document Types

Edit `DocumentExtractor.SUPPORTED_EXTENSIONS` in `0_load_db.py`:

```python
SUPPORTED_EXTENSIONS = {
    '.txt': 'extract_text',
    '.your_ext': 'your_custom_method',
    # ...
}
```

Then implement the extraction method:

```python
@staticmethod
def extract_your_format(file_path: str) -> str:
    # Your extraction logic
    return text
```

### Customizing Agent Behavior

Modify the `moderation_section` in agent files to adjust:
- Token usage limits (currently 75,000 max)
- Search result limits
- Response verbosity
- Tool usage priorities

---

## 📊 Performance Notes

- **Embedding Model**: `all-MiniLM-L6-v2` (384 dimensions, fast, offline)
- **Chunk Size**: ~150 tokens optimal for this model
- **Vector Search**: Cosine similarity with HNSW index
- **Token Budget**: Agent constrained to 75k tokens per conversation

---

## ⚠️ Important Considerations

1. **Token Costs**: The agent is configured for cost-conscious usage. The system prompt explicitly encourages efficiency due to "paywall prison" constraints.

2. **Data Privacy**: Documents are stored locally in `./chroma_db`. No data is sent to external services during ingestion.

3. **OCR Requirements**: For image text extraction, install Tesseract:
   - Ubuntu: `sudo apt-get install tesseract-ocr`
   - macOS: `brew install tesseract`
   - Windows: [Download installer](https://github.com/UB-Mannheim/tesseract/wiki)

4. **Model Access**: Requires Hugging Face token with access to Fireworks AI inference endpoints.

---

## 📄 License

[License](https://github.com/david-thrower/base-agent-template/blob/main/license.md)


---

## 🤝 Contributing

Contributions welcome (but keep it simple, this is meant to be a simple and quick start agent kit)! Please ensure:
- Code passes `flake8` linting
- New document extractors include error handling
- Tests are added to the CI pipeline

---

## 🐛 Troubleshooting

| Issue | Solution |
|-------|----------|
| `No module named 'pytesseract'` | Install optional OCR deps: `pip install pytesseract pillow` |
| `Collection not found` | Run `0_load_db.py` first to initialize the database |
| `HF_TOKEN not set` | Export your Hugging Face token as environment variable |
| PDF extraction fails | Install `pdfplumber` (preferred) or `PyPDF2` |
| Out of memory | Reduce `TARGET_TOKENS` or process fewer files at once |
```

