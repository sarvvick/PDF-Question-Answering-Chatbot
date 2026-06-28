# Multimodal PDF Question Answering Chatbot

A Multimodal Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and ask natural language questions. The chatbot extracts **text, images, and tables** from PDFs, indexes the content using semantic embeddings, and generates accurate answers using a locally hosted Large Language Model through Ollama.

---

## Features

- Upload and process PDF documents
- Semantic search using FAISS
- Text extraction from PDFs
- Image extraction and description
- Table extraction from PDF pages
- Local LLM inference using Ollama
- Interactive question answering
- Streamlit web interface
- Runs completely offline after model download

---

## Tech Stack

| Category | Technology |
|----------|------------|
| Frontend | Streamlit |
| PDF Processing | PyMuPDF |
| Table Extraction | Camelot |
| Embeddings | Sentence Transformers |
| Vector Database | FAISS |
| LLM Framework | LangChain |
| Local LLM | Ollama (Qwen2.5-VL / LLaVA) |
| Programming Language | Python |

---

## Project Structure

```text
PDF-Question-Answering-Chatbot/
│
├── app.py                  # Main Streamlit application
├── requirements.txt
├── README.md
├── temp_images/            # Extracted images from PDFs
└── vector_store/           # FAISS vector database
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/PDF-Question-Answering-Chatbot.git

cd PDF-Question-Answering-Chatbot
```

### 2. Create Virtual Environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Install Ollama

Download and install Ollama:

https://ollama.com/download

Pull a Vision Language Model:

```bash
ollama pull qwen2.5vl
```

or

```bash
ollama pull llava
```

Verify installation:

```bash
ollama list
```

---

## Running the Application

Start Ollama:

```bash
ollama serve
```

Run the Streamlit application:

```bash
streamlit run app.py
```

Open your browser:

```text
http://localhost:8501
```

---

## Workflow

1. Upload a PDF document.
2. Extract text from all pages.
3. Extract images and generate image descriptions using a Vision LLM.
4. Extract tables from PDF pages.
5. Split content into chunks.
6. Generate embeddings using Sentence Transformers.
7. Store embeddings in a FAISS vector database.
8. Retrieve relevant chunks based on the user query.
9. Generate answers using a local LLM through Ollama.

---

## Working Screenshots

<img width="1802" height="499" alt="Screenshot 2026-06-28 105007" src="https://github.com/user-attachments/assets/fcc69d1c-e2a1-4a6c-95d4-85f641f5704e" />
<img width="1812" height="713" alt="Screenshot 2026-06-28 110259" src="https://github.com/user-attachments/assets/6b0ab9be-bffb-4a27-b1d6-46493431c6af" />

---

## Main Dependencies

```text
streamlit
langchain
langchain-community
langchain-ollama
sentence-transformers
faiss-cpu
pymupdf
camelot-py
pandas
pillow
torch
torchvision
```

---

## System Requirements

- Python 3.11+
- 8 GB RAM minimum
- 16 GB RAM recommended
- Windows / Linux / macOS
- Ollama installed locally
