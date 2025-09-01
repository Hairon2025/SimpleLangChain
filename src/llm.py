# src/llm.py
from langchain_community.chat_models.tongyi import ChatTongyi
import os

def get_llm(model_name="qwen-plus", temperature=0.7):
    if not os.getenv("DASHSCOPE_API_KEY"):
        raise ValueError("请设置 DASHSCOPE_API_KEY 环境变量")
    
    return ChatTongyi(model=model_name, temperature=temperature)