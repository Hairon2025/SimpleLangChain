from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
from langchain_community.chat_models.tongyi import ChatTongyi  # 通义千问聊天模型

# 创建一个ChatTongyi实例
llm = ChatTongyi(
    model="qwen-plus",
    api_key="sk-0ad166a422184f00aa7338de03abd122",
    temperature=0.7,
)

# 增加示例组
examples = [
    {"input": "2 + 2", "output": "4"},
    {"input": "2 + 3", "output": "5"},
]

# 构造提示词模板
example_prompt = ChatPromptTemplate.from_messages(
    [
        ("human", "{input}"),
        ("ai", "{output}"),
    ]
)

# 组合示例与提示词
few_shot_prompt = FewShotChatMessagePromptTemplate(
    example_prompt=example_prompt,
    examples=examples,
)

# 打印提示词模板
print(few_shot_prompt.invoke({}).to_messages())
print('======')

# 最终提示词
final_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "你是一位神奇的数学奇才"),
        few_shot_prompt,
        ("human", "{input}"),
    ]
)

# 重新提问
chain = final_prompt | llm
result = chain.invoke({"input": "What is 2 + 9?"})
print(result.content)