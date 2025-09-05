from langchain_community.chat_models.tongyi import ChatTongyi  # 通义千问聊天模型
from pydantic import BaseModel, Field # Pydantic数据模型定义
from typing import TypedDict, Annotated, Optional

# 初始化通义千问大模型（使用qwen-plus模型）
llm = ChatTongyi(
    model="qwen-plus",
    api_key="sk-0ad166a422184f00aa7338de03abd122",
    temperature=0.7,
)


class Joke(TypedDict):
    """
    定义一个笑话的数据模型，包含笑话的铺垫、笑点和评分。
    """
    setup: Annotated[str, ..., "The setup of the joke"]  # 笑话的铺垫部分
    punchline: Annotated[str, ..., "The punchline of the joke"]  # 笑话的笑点部分
    rating: Annotated[Optional[int], None, "How funny the joke is, from 1 to 10"]

# 注释掉的部分是用于结构化输出的例子
# structured_llm = llm.with_structured_output(Joke)  # 将llm配置为返回Joke类型的结构化输出
# output = structured_llm.invoke("给我讲一个关于程序员的笑话").content  # 调用llm并获取结构化的笑话输出

structured_llm = llm.with_structured_output(Joke)  # 将llm配置为返回Joke类型的结构化输出

for chunk in structured_llm.stream("给我讲一个关于程序员的笑话 300 字"):
    print(chunk)

# print(output) 


# 直接调用llm获取笑话文本
# llm.invoke("给我讲一个关于程序员的笑话")  # 向llm发送请求，要求生成一个关于程序员的笑话