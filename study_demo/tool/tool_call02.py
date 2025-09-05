from langchain_core.tools import tool  # 导入tool装饰器
from langchain_community.chat_models.tongyi import ChatTongyi  # 通义千问聊天模型

# 使用@tool装饰器定义乘法函数
@tool
def multiply(a: int, b: int) -> int:
    """
    Multiply two numbers.
    """
    return a * b  # 返回两个整数的乘积

# 打印乘法函数的相关信息
print(multiply.name)  # 打印函数名称
print(multiply.description)  # 打印函数描述
print(multiply.args)  # 打印函数参数信息
