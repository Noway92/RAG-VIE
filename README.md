# 🧠 RAG - Retrieval-Augmented Generation for Job Offers

## 📘 Project Overview
This project implements a **Retrieval-Augmented Generation (RAG)** workflow that retrieves, processes, and analyzes job offers from the **Civiweb API**.  
It leverages **OpenAI’s API** to enhance or summarize the collected data, enabling smarter insights or automated Q&A over job postings.

---

## 🚀 Main Features
- **API Data Collection:**  
  Fetches recent job offers from the Civiweb API using structured JSON POST requests.
  
- **Offer Details Retrieval:**  
  Retrieves detailed information for each offer by its unique ID.

- **Data Handling:**  
  Handles pagination, data structuring, and filtering for later processing.

- **OpenAI Integration:**  
  Uses the OpenAI API to process or summarize job offer data — part of the RAG (Retrieval-Augmented Generation) pipeline.
  
---

## 🏗️ Architecture Overview

### 1. **Data Retrieval Layer**
- **Civiweb API Integration**: Scripts (`BDD.py`, `Last_Refresh.py`) fetch job offers via JSON POST requests.
- **Pagination Handling**: Ensures all available offers are retrieved, even across multiple pages.

### 2. **Data Processing Layer**
- **Data Structuring**: Offers are parsed, filtered, and stored in a structured format (currently `.npz` files).
- **Embeddings Generation**: Job offer data is transformed into vector embeddings for semantic search (see `vie_embeddings.npz`).

### 3. **Storage Layer**
- **Current Storage**: Embeddings and metadata are stored in `.npz` files (e.g., `vie_embeddings.npz`).
- **Future Storage**: Migration planned to **ChromaDB** for efficient vector storage and retrieval.

### 4. **Retrieval-Augmented Generation (RAG) Layer**
- **OpenAI Integration**: Uses embeddings to retrieve relevant job offers and augment responses via OpenAI’s API.
- **Semantic Search**: Enables querying job offers based on meaning, not just keywords.

### 5. **Utility Scripts**
- **Cache Management**: `_pycache_` and `.gitignore` ensure clean project maintenance.
- **Configuration**: `RAG.ipynb` and `requirements.txt` provide setup and usage guidelines.

---

---

## 🧰 Technologies Used
- **Python 3**
- **Requests** – API interaction  
- **NumPy** – data manipulation  
- **OpenAI API** – text generation and retrieval augmentation  
- **JSON / OS** – data and environment management

---

## 💡 Possible Applications
- Automated **job offer aggregation** and **summarization**
- Building a **semantic search** or **chatbot** for job offers
- Generating structured datasets for analysis or visualization

---

## ⚙️ How to Run

1. **Install dependencies:**
   ```bash
   pip install openai requests numpy chromadb

export OPENAI_API_KEY="your_api_key"


##  Next Tasks

Use a Chroma database rather that .npz files for data storage.
