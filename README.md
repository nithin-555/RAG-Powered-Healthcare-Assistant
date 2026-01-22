# Healthcare RAG Assistant

**Retrieval • Rerank • Respond**

This project builds a **Retrieval-Augmented Generation (RAG)** healthcare assistant that retrieves trusted medical Q&A data, semantically reranks relevant content, and generates grounded responses using a large language model — all with clear source attribution. It is built for *trustworthiness and real-world reliability*, especially in healthcare domains where misinformation can be harmful. :contentReference[oaicite:0]{index=0}

---

## 🚀 Overview

Traditional AI chatbots can hallucinate or guess when they don’t know — risky for healthcare.  
This system is **designed to answer questions based on real, authoritative medical content** and to clearly cite sources so that users and developers can verify responses. :contentReference[oaicite:1]{index=1}

---

## 🔍 How It Works

When a user asks a question:

1. **Retrieve** relevant medical Q&A from a vector index built from the MedQuAD dataset.
2. **Semantic Reranking** occurs using a Cross-Encoder model to improve relevance.
3. **Structured Prompting** bundles the top relevant responses into a prompt.
4. **LLM Answer Generation** produces a grounded, natural response using Google Gemini.
5. **Source Attribution** is included in the final output so users can see where the answer came from. :contentReference[oaicite:2]{index=2}

---

## 📦 Features

✔ Grounded medical answers based on real medical Q&A  
✔ Vector search with FAISS for efficient retrieval  
✔ Semantic reranking for more accurate relevance  
✔ Large language model generation (Google Gemini)  
✔ Trusted source links in final outputs  
✔ Designed for transparent, reproducible results :contentReference[oaicite:3]{index=3}

---

## 🧠 Dataset

The system is powered by the **MedQuAD dataset** — a collection of curated medical questions and answers from trusted sources like NIH, MedlinePlus, and GARD. The dataset is preprocessed into structured dictionaries with:

- `question`
- `answer`
- `source`
- `filename`
- `url` :contentReference[oaicite:4]{index=4}

---

## 🛠️ Technologies Used

| Component | Technology |
|-----------|------------|
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) |
| Vector Store | FAISS |
| Semantic Reranking | CrossEncoder (`ms-marco-MiniLM-L-6-v2`) |
| LLM | Google Gemini (`gemini-1.5-pro`) |
| Backend | Python (pandas, pickle, dotenv) | :contentReference[oaicite:5]{index=5}

---

## 📁 Project Structure

| Folder/File | Purpose |
|-------------|---------|
| `main.py` | Entrypoint — runs the assistant |
| `utils/` | Helper utilities |
| `assets/` | Images and visual assets |
| `data_loader.py` | Loads and structures MedQuAD |
| `embeddings.py` | Builds and stores vector embeddings |
| `retriever.py` | Handles retrieval + reranking |
| `prompt_builder.py` | Formats prompts for LLM |
| `gemini_client.py` | Interacts with Google Gemini | :contentReference[oaicite:6]{index=6}

---

## 🔧 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/RaviKunapareddy/rag-healthcare-assistant.git
   cd rag-healthcare-assistant
Create & activate a virtual environment

2. **python3 -m venv venv**
source venv/bin/activate   # Windows: venv\Scripts\activate


3.**Install dependencies**
pip install -r requirements.txt


4.**Configure environment variables**
Add API keys and config values in a .env file (e.g. Gemini model keys)

5.**Run the assistant**
python main.py
