from langchain_community.chat_models.tongyi import ChatTongyi  # 通义千问聊天模型
from langchain_core.prompts import PromptTemplate
import os
from langchain_core.output_parsers import BaseOutputParser

# 自定义类
class CommaSeparatedListOutputParser(BaseOutputParser):
    """Parse the output of an LLM call to a comma-separated list."""

    def parse(self, text: str):
        """Parse the output of an LLM call."""
        print(text)
        return text.strip().split(",")

# 初始化通义千问大模型（使用qwen-plus模型）
llm = ChatTongyi(
    model="qwen-plus",
    api_key="sk-0ad166a422184f00aa7338de03abd122",
    temperature=0.7,
)

# 定义提示模板
prompt = PromptTemplate.from_template("你是一个起名大师, 请模仿示例起3个具有{county}特色的名字, 示例：男孩常用名{boy}, 女孩常用名{girl}。请返回以逗号分隔的列表形式。仅返回逗号分隔的列表，不要返回其他内容。")

# 格式化消息
message = prompt.format(county="美国", boy="sam", girl="lucy")
print(message)

# 调用模型并获取响应
text_content = llm.invoke(message).content

# 创建自定义解析器实例并解析模型的响应
parser = CommaSeparatedListOutputParser()
names = parser.parse(text_content)
print(names)