# 纳美星语网址 https://learnnavi.org/
from typing import Iterable
from langchain_community.chat_models.tongyi import ChatTongyi  # 通义千问聊天模型
from langchain_core.messages import AIMessage, AIMessageChunk
import re
import os

model = ChatTongyi(
    model="qwen-plus",
    api_key="sk-0ad166a422184f00aa7338de03abd122",
    # temperature=0,
)


def parse(ai_message: AIMessage) -> str:
    """Convert message to Na'vi style language."""
    content = ai_message.content

    # 纳美语特征转换规则
    navi_rules = {
        # 常见问候语转换
        r'\b[Hh]ello\b': 'Kaltxi',
        r'\b[Hh]i\b': 'Kaltxi',
        r'\b[Tt]hank you\b': 'Irayo',
        r'\b[Gg]oodbye\b': 'Eywa ngahu',

        # 添加纳美语特征的后缀
        r'\b(friend|friends)\b': 'eylan',
        r'\b(brother|brothers)\b': 'tsmukan',
        r'\b(sister|sisters)\b': 'tsmuke',

        # 添加纳美语语气词
        r'[.!]\s*$': ' kxe!',
        r'\?\s*$': ' srak?',

        # 在某些词前添加特征前缀
        r'\b(beautiful|pretty)\b': 'na\'vi',
        r'\b(sacred|holy)\b': 'txe\'lan',
    }

    # 应用转换规则
    result = content
    for pattern, replacement in navi_rules.items():
        result = re.sub(pattern, replacement, result)

    # 添加纳美语特征词缀
    result = f"Oel ngati kameie... {result}"

    return result

chain = model | parse
print(chain.invoke("hello"))