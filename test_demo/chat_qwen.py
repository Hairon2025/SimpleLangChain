import getpass
import os
from langchain_community.chat_models.tongyi import ChatTongyi  # 通义千问聊天模型
from langchain_core.messages import HumanMessage, AIMessage

# 设置DashScope API密钥（阿里云千问所需）
if not os.environ.get("DASHSCOPE_API_KEY"):
    print("请输入你的 DASHSCOPE API KEY（输入时不会显示字符）：")
    # 如果环境变量中没有密钥，通过安全输入获取
    os.environ["DASHSCOPE_API_KEY"] = getpass.getpass()
    
# 初始化通义千问大模型（使用qwen-plus模型）
llm = ChatTongyi(model="qwen-plus")

# Step 3: 支持多轮对话的历史记录
messages = []

print("\n🤖 欢迎使用通义千问助手！输入 'quit' 或 'exit' 退出，输入 'clear' 清除历史。\n")

# Step 4: 循环提问
while True:
    try:
        user_input = input("👤 你: ").strip()
        
        if user_input.lower() in ['quit', 'q', 'Q', 'exit']:
            print("👋 再见！")
            break
        elif user_input.lower() == 'clear':
            messages.clear()
            print("🧹 已清除对话历史。\n")
            continue
        elif not user_input:
            print("⚠️ 请输入有效问题。\n")
            continue

        # 添加用户消息到历史
        messages.append(HumanMessage(content=user_input))

        # 调用模型生成回复（传入完整对话历史）
        response = llm.invoke(messages)

        # 提取回复内容
        ai_message = response.content
        print(f"🤖 Qwen: {ai_message}\n")

        # 将 AI 回复也加入历史，支持上下文理解
        messages.append(AIMessage(content=ai_message))

    except KeyboardInterrupt:
        print("\n\n👋 检测到退出信号，再见！")
        break
    except Exception as e:
        print(f"❌ 发生错误: {e}")
        print("提示：检查网络连接或 API Key 是否正确。\n")