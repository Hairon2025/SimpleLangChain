# 使用OutputFixingParser进行错误修复
from langchain.output_parsers import OutputFixingParser
from typing import List
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from langchain_community.chat_models.tongyi import ChatTongyi  # 通义千问聊天模型
import os

# 定义一个演员模型，有两个字段
class Actor(BaseModel):
    name: str = Field(description="name of an actor")
    film_names: List[str] = Field(description="list of names of films they starred in")

actor_query = "Generate the filmography for a random actor."
parser = PydanticOutputParser(pydantic_object=Actor)

# 假设生成的错误值
misformatted = "{'name': 'Tom Hanks', 'film_names': ['Forrest Gump']}"

# 运行的时候抛出错误
# try:
#     parser.parse(misformatted)
# except OutputParserException as e:
#     print(e)

# 使用OutputFixingParser可以修复错误
# 定义修复所依赖的LLM
new_parser = OutputFixingParser.from_llm(
    parser=parser,
    llm=ChatTongyi(
        model="qwen-plus",
        api_key="sk-0ad166a422184f00aa7338de03abd122",
        temperature=0,
    ),
)
# 传入报错信息

print(new_parser.parse(misformatted))