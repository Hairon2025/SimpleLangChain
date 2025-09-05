# 引入依赖包，分别用于XML输出解析、提示词模板创建、DeepSeek模型调用以及环境变量操作
from langchain_core.output_parsers import XMLOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi  # 通义千问聊天模型
import os

# 创建一个ChatTongyi实例
llm = ChatTongyi(
    model="qwen-plus",
    api_key="sk-0ad166a422184f00aa7338de03abd122",
    temperature=0,
)
# 定义查询请求，要求生成汤姆·汉克斯的精简作品年表
actor_query = "Generate the shortened filmography for Tom Hanks."

# 实例化XML输出解析器，用于将模型输出解析为XML格式
# parser = XMLOutputParser()

# 以下是待添加到提示词中的指令相关代码（目前注释掉，可根据需求启用）
parser = XMLOutputParser(tags=["movies", "actor", "film", "name", "genre"])  # 可指定XML解析的标签
parser.get_format_instructions()  # 获取解析器的格式说明指令

# 创建提示词模板，模板包含查询内容和解析器的格式说明
prompt = PromptTemplate(
    template="""{query}\n{format_instructions}""",  # 提示词模板内容，将查询和格式说明拼接
    input_variables=["query"],  # 声明提示词中需要动态输入的变量为"query"
    # 声明部分变量为固定值，这里是解析器的格式说明，由parser.get_format_instructions()获取
    partial_variables={"format_instructions": parser.get_format_instructions()},
)

# 使用LangChain的表达式语言，将提示词、大模型、解析器按顺序组成一个链
chain = prompt | llm | parser
# 调用这个链，传入查询参数，执行整个流程并获取输出结果
output = chain.invoke({"query": actor_query})
# 打印最终的输出结果
print(output)