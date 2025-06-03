#  IEP Goal Generator using RAG and LLM

This project is a Retrieval-Augmented Generation (RAG) system designed to assist special education professionals in writing compliant, individualized IEP transition goals. It uses a language model (Zephyr-7B or similar) grounded in real-world educational standards, career information, and IDEA 2004 compliance guidelines.

---

##  Project Structure

| File                | Description |
|---------------------|-------------|
| `app.py`            | A Streamlit web application that allows users to enter a student profile and generate IEP transition goals using the underlying RAG system. |
| `load_and_embed.py` | Script that collects, cleans, chunks, and embeds source documents (like state standards and career info). The data is stored in a local vector database (`rag_vector_store`) using Chroma. Run this once before using the app. |
| `retriever.py`      | A module that retrieves relevant information from the vector store based on the user’s input (i.e., the student profile). It powers the “retrieved_context” used in the generation prompt. |
| `generator.py`      | The module responsible for generating IEP goals using a language model (either locally or via Hugging Face Inference API). It formats and sends a prompt that includes the student profile and retrieved documents. |
| `requirements.txt`  | Lists all Python dependencies required for this project, including LangChain, Streamlit, Hugging Face Transformers, Chroma, PDF/HTML parsing, and more. Use this file to install everything with `pip install -r requirements.txt`. |
| `RAG Pipeline for an IEP Final.ipynb`  | Steps through and explains each step of the pipeline from Data Collection and Preprocessing, RAG Implementation, and Prompting to User Interface deployment, |

---

## `rag_vector_store/`

Because this folder contains a model object, it was not included in this repository.  However, when this folder is generated it contains the **persistent vector database** created by `load_and_embed.py` using Chroma. It includes:

- Document chunks from sources like:
  - The Occupational Outlook Handbook
  - State 21st Century Skills standards
  - IEP goal templates and documentation
- Embeddings generated using a model like `all-MiniLM-L6-v2`
- FAISS index + metadata for fast similarity search

> This folder is automatically generated the first time you run `load_and_embed.py`. It should exist before running the Streamlit app.

---

##  How to Use
Before following the below steps, enter your hugging face token code in the app.py file.  Also - update it in the Jupyter notebook RAG Pipeline for An IEP Final, if you want to step through the code in that notebook.

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run preprocessing (first time only)

```bash
python load_and_embed.py
```

### 3. Start the app

```bash
streamlit run app.py
```
