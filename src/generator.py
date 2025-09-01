# src/generator.py
from langchain import hub

def get_rag_prompt(prompt_name="rlm/rag-prompt"):
    return hub.pull(prompt_name)