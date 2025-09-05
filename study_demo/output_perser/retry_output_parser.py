# 导入依赖包，用于输出解析、异常处理、提示词模板、OpenAI模型调用和数据模型定义
from langchain.output_parsers import RetryOutputParser
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi  # 通义千问聊天模型
from pydantic import BaseModel, Field

# 初始化通义千问大模型（使用qwen-plus模型）
llm = ChatTongyi(
    model="qwen-plus",
    api_key="sk-0ad166a422184f00aa7338de03abd122",
    temperature=0,
)

# 定义提示词模板内容，用于指导模型生成Action和Action Input
template = """Based on the user question, provide an Action and Action Input for what step should be taken.
{format_instructions}
Question: {query}
Response:"""

# 定义Action数据模型，包含action和action_input两个字段
class Action(BaseModel):
    action: str = Field(description="action to take")
    action_input: str = Field(description="input to the action")

# 实例化PydanticOutputParser，指定要解析到的Pydantic对象为Action
parser = PydanticOutputParser(pydantic_object=Action)

# 创建提示词模板，用于生成模型输入的提示词
prompt = PromptTemplate(
    template="Answer the user query.\n{format_instructions}\n{query}\n",
    input_variables=["query"],
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 假设的用户输入，合成提示值
prompt_value = prompt.format_prompt(query="北京今天天气如何?")

# 假设得到的一个错误回答，不符合Pydantic的字段要求（缺少action_input字段）
bad_response = '{"action": "search"}'

# # 以下是尝试解析错误响应，会抛出异常的代码（目前注释掉）
# try:
#     parser.parse(bad_response)
# except OutputParserException as e:
#     print(e)

# 使用RetryOutputParser实现错误重试，定义使用的模型进行重试
retry_parser = RetryOutputParser.from_llm(
    parser=parser,
    llm=llm
)

# 传入错误信息以及原始的提示值，进行重试解析
print(retry_parser.parse_with_prompt(bad_response, prompt_value))