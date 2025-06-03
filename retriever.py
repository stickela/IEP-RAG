from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings

def retrieve_relevant_chunks(query, persist_dir="rag_vector_store", k=5):
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory=persist_dir, embedding_function=embedding_model)
    docs = db.similarity_search(query, k=k)
    return "\n\n".join([doc.page_content for doc in docs])