# 🔍 Visual-First Financial Document Intelligence Agent

> **Multimodal RAG system that uses Computer Vision + LLMs to extract and query financial data from complex PDFs**

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.52-FF4B4B.svg)](https://streamlit.io)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.12-000000.svg)](https://llamaindex.ai)
[![OpenRouter](https://img.shields.io/badge/OpenRouter-AI-purple.svg)](https://openrouter.ai)

---

## 🎯 Problem Statement

Financial analysts spend **hours** manually cross-referencing data between narrative text and tables in documents like 10-K filings. Traditional OCR solutions fail because they:

- Treat documents as plain text (losing table structure)
- Can't handle complex layouts with charts and multi-column formats
- Don't understand financial context or "read" tables visually

## 💡 Solution

A **Vision-First RAG Pipeline** that:

1. **Ingests** financial PDFs (10-K filings).
2. **Extracts** structured data using **GPT-4o-mini** (via OpenRouter) and strict PDF parsing.
3. **Summarizes** tables visually using Vision-Language Models.
4. **Indexes** visual and textual content into a local vector database.
5. **Answers** natural language queries and **shows the actual source table image** for verification.

---

## 🏗️ Architecture

```
┌─────────────┐
│   PDF File  │
└──────┬──────┘
       │
       ▼
┌─────────────────────┐
│  Ingestion Engine   │
│  (src/rag/ingest)   │
│  • PDF Parsing      │
│  • Vision Summarizer│
└──────┬──────────────┘
       │ Text Chunks + Table Summaries
       ▼
┌─────────────────────┐
│  Vector Database    │  ← LlamaIndex + Embeddings
│  (Local Storage)    │
└──────┬──────────────┘
       │
       ▼
┌─────────────────────┐       ┌─────────────────┐
│  Query Engine       │ ◄──── ►  OpenRouter LLM │
│  (src/rag/query)    │       │ (GPT-4o/Gemini) │
└──────┬──────────────┘       └─────────────────┘
       │
       ▼
┌─────────────────────┐
│  Chat Interface     │  ← Streamlit UI
│  (app.py)           │
└─────────────────────┘
```

---

## 🚀 Quick Start

### 1. Prerequisites

- Python 3.12+
- `uv` (recommended) or `pip`
- OpenRouter API Key

### 2. Installation

```bash
git clone https://github.com/Mukundh0007/agentic-rag.git
cd agentic-rag

# Install dependencies with uv (fastest)
uv sync

# Or with pip
pip install -r requirements.txt
```

### 3. Configuration

Create a `.env` file in the root directory:

```env
OPENROUTER_API_KEY=sk-or-v1-xxxxxxxx...
```

### 4. Usage

We provide a central controller `main.py` for most tasks, but the computer vision pipeline requires initialization.

**Step 1: Setup Models**
Download YOLOv8 weights and the fine-tuned table detector.

```bash
uv run python src/download_weights.py
```

**Step 2: Extract Tables (Computer Vision)**
To verify the environment, run:

```bash
uv run python src/verify.py
```

Then, run the YOLOv8 pipeline to crop tables from your PDF.
```bash
uv run python src/vision/vision_processor.py
```

*(This saves images to `data/processed_tables/`)*

**Step 3: Ingest Data**
Process the PDF text and the extracted table images into the vector index.

```bash
uv run python main.py --ingest
```

**Step B: Launch Web App (The "Wow" Factor)**
Start the visual chat interface.

```bash
uv run python main.py --app
```

**Step C: CLI Query (Optional)**
Run a quick test query from the terminal.

```bash
uv run python main.py --query "What are the primary risk factors?"
```

---

## 📂 Project Structure

```
agentic-rag/
├── 📄 main.py                      # 🎮 Central CLI controller
├── 📄 app.py                       # 🖥️ Streamlit Web Application
├── 📄 pyproject.toml               # Dependency configuration
├── 📄 requirements.txt             # Pip requirements
├── 📄 README.md                    # Documentation
│
├── 📂 src/
│   └── 📂 rag/
│       ├── 📄 ingest.py            # 🏗️ Ingestion pipeline (PDF -> Vector DB)
│       ├── 📄 query.py             # 🔍 Retrieval & Query logic
│       └── 📄 openrouter_client.py # 🔌 Custom LlamaIndex adapter for OpenRouter
│
├── 📂 data/
│   ├── 📄 apple_10k.pdf            # Source Document
│   └── 📂 processed_tables/        # Extracted table images
│
└── 📂 storage/                     # Local Vector Store (created after ingest)
```

---

## ✨ Key Features

- **Robust PDF Parsing**: Uses `PDFReader` for accurate text extraction (no garbage binary text).
- **Visual Verification**: The chatbot displays the **actual images** of the tables it used to answer your question.
- **Smart Routing**: `main.py` handles CLI commands and app launching seamlessly.
- **Cost Effective**: Optimized to use efficient models like `gpt-4o-mini` via OpenRouter.

---

## 📊 Example Interaction

**User Query**: *"What was the total net sales in 2024?"*

**AI Response**:
> Apple's total net sales in 2024 were **$391.04 billion**.

**Verified Sources**:

- `p23_table_5.png` (Shows the Income Statement)
- `p32_table_13.png` (Shows Segment Breakdown)

*(The UI displays these images automatically)*

---

## 📝 License

MIT License
