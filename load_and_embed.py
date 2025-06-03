# load_and_embed.py

import os
import aiohttp
import asyncio
import argparse
import pdfplumber
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor

from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

# ---- Configurable Parameters ---- #
PERSIST_DIR = "rag_vector_store"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

SOURCES = {
    "Occupational Outlook Handbook": "https://www.bls.gov/ooh/",
    "State Guidelines": "https://educate.iowa.gov/pk-12/standards/academics/21st-century-skills/",
    "Transition Planning documentation": "https://educate.iowa.gov/media/8593/download?inline",
    "IEP Components": "https://iowaideainformation.org/special-education/individualized-education-programs/components-of-an-iep/",
    "Sample IEP Goals": "https://www.powerschool.com/blog/how-to-write-appropriate-and-achievable-iep-goals/"
}

# ---- Helper Functions ---- #

def extract_text_from_pdf(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        return "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())

async def fetch_text_from_url(session, name, url):
    async with session.get(url) as resp:
        html = await resp.text()
        soup = BeautifulSoup(html, 'html.parser')
        return Document(page_content=soup.get_text(separator="\n", strip=True), metadata={"source": name})

async def download_all_sources():
    tasks, docs = [], []
    async with aiohttp.ClientSession() as session:
        for name, url in SOURCES.items():
            if url.endswith(".pdf"):
                pdf_path = f"{name.replace(' ', '_')}.pdf"
                if not os.path.exists(pdf_path):
                    content = await session.get(url)
                    with open(pdf_path, "wb") as f:
                        f.write(await content.read())
                text = extract_text_from_pdf(pdf_path)
                docs.append(Document(page_content=text, metadata={"source": name}))
            else:
                tasks.append(fetch_text_from_url(session, name, url))
        docs.extend(await asyncio.gather(*tasks))
    return docs

def parallel_chunk_documents(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    with ThreadPoolExecutor() as executor:
        chunks = list(executor.map(splitter.split_documents, [[doc] for doc in docs]))
    return [chunk for group in chunks for chunk in group]

def load_and_embed_documents(force_rebuild=False):
    if os.path.exists(os.path.join(PERSIST_DIR, "index")) and not force_rebuild:
        print(f"Vector store already exists at '{PERSIST_DIR}'. Use --force to rebuild.")
        return

    print("Downloading and extracting documents...")
    documents = asyncio.run(download_all_sources())

    print("Chunking documents in parallel...")
    chunks = parallel_chunk_documents(documents)

    print(f"Generating embeddings with model: {EMBEDDING_MODEL_NAME} ...")
    embedding_model = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vectorstore = Chroma.from_documents(chunks, embedding_model, persist_directory=PERSIST_DIR)

    print("Saving vector store...")
    vectorstore.persist()

    print(f"Successfully stored {len(chunks)} chunks at '{PERSIST_DIR}'")

# ---- CLI Entrypoint ---- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load, process, and embed special education IEP source documents.")
    parser.add_argument("--force", action="store_true", help="Force rebuild of the vector store even if it exists")

    args = parser.parse_args()
    load_and_embed_documents(force_rebuild=args.force)
