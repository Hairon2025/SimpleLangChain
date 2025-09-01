# src/loader.py
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from bs4 import SoupStrainer

def load_docs_from_config(config):

    urls = config["web"]["urls"]
    bs_kwargs = config["web"].get("bs_kwargs", {})

    # 解析 class 过滤
    if "parse_only" in bs_kwargs:
        parse_only = SoupStrainer(class_=bs_kwargs["parse_only"]["class"])
        bs_kwargs = {"parse_only": parse_only}

    loader = WebBaseLoader(web_paths=urls, bs_kwargs=bs_kwargs)
    docs = loader.load()

    # 分割文本
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config["text_splitter"]["chunk_size"],
        chunk_overlap=config["text_splitter"]["chunk_overlap"],
    )
    return splitter.split_documents(docs)