# app/cli.py
import os
import getpass
from src.loader import load_docs_from_config
from src.vectorstore import get_vectorstore
from src.llm import get_llm
from src.generator import get_rag_prompt
from src.graph import create_rag_graph
import yaml

def main():
    # 加载配置
    with open("./config/settings.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 设置 API Key
    if not os.environ.get("DASHSCOPE_API_KEY"):
        print("🔐 请输入 DASHSCOPE API Key:")
        os.environ["DASHSCOPE_API_KEY"] = getpass.getpass()

    # 初始化组件
    print("🚀 初始化向量数据库...")
    vectorstore = get_vectorstore(config["vectorstore"]["persist_directory"])

    # 如果数据库为空，导入数据
    if not vectorstore.get()["documents"]:
        print("📥 向量库为空，正在加载网页内容...")
        splits = load_docs_from_config(config)
        vectorstore.add_documents(splits)
        print("✅ 数据已导入向量库")
    else:
        print("🔁 使用已有向量库")

    llm = get_llm(config["model"]["llm"], config["model"]["temperature"])
    prompt = get_rag_prompt(config["rag"]["prompt"])
    graph = create_rag_graph(vectorstore, llm, prompt)

    # 开始交互
    print("\n💬 欢迎使用 RAG 助手！输入 'quit' 退出。\n")
    while True:
        try:
            question = input("❓ 问题: ").strip()
            if question.lower() in ["quit", "q", "Q", "exit"]:
                break
            if not question:
                continue
            result = graph.invoke({"question": question})
            print(f"✅ 答案: {result['answer']}\n")
        except KeyboardInterrupt:
            break
    print("\n👋 再见！")

if __name__ == "__main__":
    main()