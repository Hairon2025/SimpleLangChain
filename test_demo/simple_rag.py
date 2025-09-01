import getpass
import os
from typing import List, TypedDict  # 用于类型注解
from langchain_core.documents import Document  # LangChain文档类型
from langchain_community.chat_models.tongyi import ChatTongyi  # 通义千问聊天模型
from langchain_community.embeddings import DashScopeEmbeddings  # 阿里云DashScope嵌入模型
from langchain_community.document_loaders import WebBaseLoader  # 网页内容加载器
from langchain_text_splitters import RecursiveCharacterTextSplitter  # 文本分割器
from langchain_chroma import Chroma  # Chroma向量数据库
from langgraph.graph import START, StateGraph  # LangGraph状态图
from langchain import hub  # LangChain提示词中心
import bs4  # 网页解析库

# 设置DashScope API密钥（阿里云千问所需）
if not os.environ.get("DASHSCOPE_API_KEY"):
    # 如果环境变量中没有密钥，通过安全输入获取
    os.environ["DASHSCOPE_API_KEY"] = getpass.getpass()

# 初始化通义千问大模型（使用qwen-plus模型）
llm = ChatTongyi(model="qwen-plus")

# ==================== 数据加载与处理 ==================== 
# 使用WebBaseLoader加载指定网页内容
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        # 通过BeautifulSoup仅解析特定class的内容（提高效率）
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()  # 执行加载操作

# 使用递归字符文本分割器处理文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # 每个文本块大小
    chunk_overlap=200  # 块之间重叠字符数（保持上下文）
)
all_splits = text_splitter.split_documents(docs)  # 执行分割操作

# ==================== 向量存储与检索 ==================== 
# 初始化DashScope的文本嵌入模型（用于生成向量）
embeddings = DashScopeEmbeddings(model="text-embedding-v2")

# 创建Chroma向量数据库实例
vector_store = Chroma(
    collection_name="example_collection",  # 集合名称
    embedding_function=embeddings,  # 使用的嵌入模型
    persist_directory="./chroma_langchain_db"  # 本地持久化目录
)

# 将分割后的文档添加到向量数据库
_ = vector_store.add_documents(documents=all_splits)

# ==================== 提示词与状态定义 ==================== 
# 从LangChain Hub获取预定义的RAG提示词模板
prompt = hub.pull("rlm/rag-prompt")

# 定义应用的状态结构（类型化字典）
class State(TypedDict):
    question: str    # 用户问题
    context: List[Document]  # 检索到的文档上下文
    answer: str      # 生成的答案

# ==================== 定义处理步骤 ==================== 
def retrieve(state: State):
    """检索阶段：根据问题从向量库查找相关文档"""
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    """生成阶段：结合上下文生成最终答案"""
    # 将检索到的文档内容合并
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    
    # 使用提示词模板格式化输入
    messages = prompt.invoke({
        "question": state["question"], 
        "context": docs_content
    })
    
    # 调用大模型生成回答
    response = llm.invoke(messages)
    return {"answer": response.content}

# ==================== 构建应用流程图 ==================== 
# 创建状态图并定义处理流程
graph_builder = StateGraph(State)
graph_builder.add_sequence([retrieve, generate])  # 添加顺序执行节点
graph_builder.add_edge(START, "retrieve")  # 设置起始节点
graph = graph_builder.compile()  # 编译流程图

# ==================== 测试应用 ==================== 
# 输入问题并获取回答
# response = graph.invoke({"question": "什么是ReAct？"})
# print("最终答案：", response["answer"])

# ==================== 测试应用：多轮提问 ==================== 
print("\n🔍 欢迎使用 RAG 助手！我可以通过网页内容回答你的问题。")
print("📌 当前知识库来自: https://lilianweng.github.io/posts/2023-06-23-agent/")
print("💡 输入 'quit' 或 'exit' 退出，输入 'clear' 清除上下文（每次问题独立检索）。\n")

while True:
    try:
        user_question = input("❓ 请输入你的问题: ").strip()
        
        # 退出命令
        if user_question.lower() in ["quit", 'q', 'Q', "exit"]:
            print("👋 再见！")
            break
            
        # 忽略空输入
        if not user_question:
            print("⚠️ 请输入有效问题。\n")
            continue
            
        # 调用图执行流程（每次独立检索 + 生成）
        print("🔄 正在检索相关信息...")
        response = graph.invoke({"question": user_question})
        answer = response["answer"]
        
        # 输出结果
        print(f"✅ 答案: {answer}\n")
        
    except KeyboardInterrupt:
        print("\n\n👋 被动中断，再见！")
        break
    except Exception as e:
        print(f"❌ 执行出错: {e}\n")