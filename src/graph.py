# src/graph.py
from langgraph.graph import StateGraph, START
from typing import TypedDict, List
from langchain_core.documents import Document
from .retriever import create_retriever
from .generator import get_rag_prompt
from .llm import get_llm

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def create_rag_graph(vectorstore, llm_model, prompt):
    retriever = create_retriever(vectorstore)
    graph_builder = StateGraph(State)

    def retrieve(state):
        docs = retriever.invoke(state["question"])
        return {"context": docs}

    def generate(state):
        context = "\n\n".join([doc.page_content for doc in state["context"]])
        messages = prompt.invoke({"question": state["question"], "context": context})
        response = llm_model.invoke(messages)
        return {"answer": response.content}

    graph_builder.add_node("retrieve", retrieve)
    graph_builder.add_node("generate", generate)
    graph_builder.add_edge(START, "retrieve")
    graph_builder.add_edge("retrieve", "generate")

    return graph_builder.compile()