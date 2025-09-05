# 引入依赖包，这里的pydantic版本为v2
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field, model_validator
from langchain_community.chat_models.tongyi import ChatTongyi  # 通义千问聊天模型
import os

# 创建一个ChatTongyi实例
llm = ChatTongyi(
    model="qwen-plus",
    api_key="sk-0ad166a422184f00aa7338de03abd122",
    temperature=0,
)

# 定义一个名为Joke的数据模型
# 必须要包含的数据字段：铺垫(setup)、抖包袱(punchline)
class Joke(BaseModel):
    setup: str = Field(description="笑话中的铺垫问题，必须以?结尾")
    punchline: str = Field(description="笑话中回答铺垫问题的部分，通常是一种抖包袱方式回答铺垫问题，例如谐音、会错意等")

    # 验证器，你可以根据自己的数据情况进行自定义
    # 注意mode=before意思是数据被转成pydantic模型的字段之前，对原始数据进行验证
    @model_validator(mode="before")
    @classmethod
    def question_ends_with_question_mark(cls, values: dict) -> dict:
        setup = values.get("setup")
        # 如果铺垫问题没有以问号结尾则抛出错误
        if setup and not setup.endswith(('?', '？')):
            raise ValueError("Badly formed question!")
        return values

# 实例化解析器、提示词模板
parser = PydanticOutputParser(pydantic_object=Joke)
# 注意，提示词模板中需要部分格式化解析器的格式要求format_instructions
prompt = PromptTemplate(
    template="回答用户的查询.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 使用LCEL语法组合一个简单的链
prompt_and_model = prompt | llm
output = prompt_and_model.invoke({"query": "给我讲一个笑话"})
print(parser.invoke(output))