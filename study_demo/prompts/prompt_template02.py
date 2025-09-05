from langchain_community.chat_models.tongyi import ChatTongyi  # 通义千问聊天模型
from langchain_core.prompts import StringPromptTemplate
import os
import inspect

# 定义一个简单的函数作为示例效果
def hello_world(abc):
    print("Hello, world!")
    return abc

# 函数大师：根据函数名称，查找函数代码，并给出中文的代码说明
PROMPT = """\
你是一个非常有经验和天赋的程序员，现在给你如下函数名称，你会按照如下格式，输出这段代码的名称、源代码、中文解释。
函数名称: {function_name}
源代码:
{source_code}
代码解释:
"""

# 获取源代码的辅助函数
def get_source_code(function_name):
    # 获得源代码
    return inspect.getsource(function_name)

# 自定义的模板类 用来实现函数大师
class CustomPrompt(StringPromptTemplate):

    def format(self, **kwargs) -> str:
        # 获得源代码和函数名
        source_code = get_source_code(kwargs["function_name"])
        
        # 生成提示词模板
        prompt = PROMPT.format(
            function_name=kwargs["function_name"].__name__,
            source_code=source_code,
            code_explanation=self._get_code_explanation(source_code)
        )
        
        return prompt

    def _get_code_explanation(self, source_code):
        # 这里可以实现具体的代码解释逻辑
        # 例如，使用自然语言处理模型来生成代码解释
        return "这是一个示例函数，用于演示如何打印 'Hello, world!' 并返回传入的参数。"

# 使用自定义的提示词模板，而不是类似对话提示词模板
a = CustomPrompt(input_variables=["function_name"])
pm = a.format(function_name=hello_world)

print(pm)


# 和LLM连接起来
# Connect to LLM
# 初始化通义千问大模型（使用qwen-plus模型）
llm = ChatTongyi(
    model="qwen-plus",
    api_key="sk-0ad166a422184f00aa7338de03abd122",
    temperature=0.7,
)

# 使用LLM的invoke方法处理之前生成的提示文本pm
msg = llm.invoke(pm)

# 打印LLM返回的消息
print(msg.content)