from pydantic import BaseModel, Field
from langchain_community.chat_models.tongyi import ChatTongyi  # 通义千问聊天模型
import os

# 定义加法工具类
class Add(BaseModel):
    """
    Add two integers.
    """
    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")

# 定义乘法工具类
class Multiply(BaseModel):
    """
    Multiply two integers.
    """
    a: int = Field(..., description="First integer")
    b: int = Field(..., description="Second integer")

# 创建一个ChatTongyi实例
llm = ChatTongyi(
    model="qwen-plus",
    api_key="sk-0ad166a422184f00aa7338de03abd122",
    temperature=0.7,
)

# 定义工具列表，包含加法和乘法工具
tools = [Add, Multiply]  # 一个模型可以绑定多个工具的

# 将工具绑定到llm
llm_with_tools = llm.bind_tools(tools)

# 定义查询问题
query = "3乘以12是多少？"

# 调用llm并获取工具调用结果
result = llm_with_tools.invoke(query).tool_calls
print(result)  # 打印工具调用结果