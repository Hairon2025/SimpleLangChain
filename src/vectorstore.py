# src/vectorstore.py
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_chroma import Chroma
import os
import shutil

def get_vectorstore(persist_dir="./chroma_db", recreate=False):
    embeddings = DashScopeEmbeddings(model="text-embedding-v2")

    if recreate and os.path.exists(persist_dir):
        shutil.rmtree(persist_dir)

    return Chroma(
        collection_name="example_collection",
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )