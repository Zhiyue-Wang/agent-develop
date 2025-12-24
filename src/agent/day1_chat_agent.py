import os
import logging
from dotenv import load_dotenv, find_dotenv, dotenv_values
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_community.chat_models.tongyi import ChatTongyi

# 1. 必须在最开头加载环境变量
load_dotenv(find_dotenv())

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("langchain_agent_debug")
# 验证 Key 是否读取成功
api_key = os.getenv("DASHSCOPE_API_KEY")
if not api_key:
    raise ValueError("未检测到 DASHSCOPE_API_KEY，请检查 .env 文件")

# 2. 定义工具（使用最新的 Pydantic v2 兼容模式）
@tool
def search_latest_info(query: str) -> str:
    """搜索最新的技术动态或资讯。"""
    # 模拟返回
    return f"关于 {query} 的搜索结果：Python 3.14 运行稳定。"

tools = [search_latest_info]

# 3. 初始化模型
# 注意：直接传入 api_key 可以绕过部分环境检测逻辑，减少报错可能
# model = ChatOpenAI(
#     model="gpt-4o",
#     api_key=api_key, # 显式传递
#     temperature=0,
#     streaming=True
# )
model = ChatTongyi(
    model="qwen-turbo",  # 千问模型版本（qwen-turbo/qwen-plus/qwen-max）
    temperature=0,  # 输出稳定，便于调试
    dashscope_api_key=api_key,  # 阿里云API Key
    verbose=True
)

# 4. 使用 LangGraph 的高级封装（最推荐的 Agent 构建方式）
# create_react_agent 是目前最稳定且兼容性最好的 Agent 生成器
agent_executor = create_react_agent(model, tools)

# 5. 执行测试函数
async def chat():
    print("--- Agent 开始运行 ---")
    inputs = {"messages": [HumanMessage(content="你好，帮我搜一下 Python 3.14 的新特性")]}
    
    # 异步迭代输出
    async for event in agent_executor.astream(inputs, stream_mode="values"):
        message = event["messages"][-1]
        if hasattr(message, "content"):
            print(f"Agent: {message.content}")

# 5. 多轮对话核心函数（异步）
async def multi_round_chat():
    """支持多轮交互式对话的核心函数，维护对话历史并处理用户输入"""
    print("=" * 60)
    print("欢迎使用千问Agent多轮对话工具！")
    print("提示：输入问题即可获取回答，输入 'exit'/'quit'/'退出' 可终止对话。")
    print("=" * 60 + "\n")

    # 初始化对话历史（保存所有消息：用户消息 + Agent回复）
    conversation_history = []

    # 无限循环接收用户输入
    while True:
        # 获取用户输入并处理空输入
        user_input = input("\n请输入你的问题：").strip()

        # 处理退出指令
        if user_input.lower() in ["exit", "quit", "退出"]:
            print("\n对话结束，再见！")
            break

        # 处理空输入
        if not user_input:
            print("⚠️  输入不能为空，请重新输入！")
            continue

        # 将用户输入添加到对话历史
        conversation_history.append(HumanMessage(content=user_input))
        print("\n----- Agent正在处理，请稍候 -----")

        try:
            # 调用Agent处理（传入最新的对话历史）
            # stream_mode="values" 表示按值流式输出，只返回最终的结果值
            async for event in agent_executor.astream(
                {"messages": conversation_history},  # 传递完整的对话历史
                stream_mode="values"
            ):
                # 获取最新的消息（Agent的回复）
                latest_message = event["messages"][-1]
                # 仅输出有内容的消息（过滤工具调用的中间消息）
                if hasattr(latest_message, "content") and latest_message.content:
                    # 将Agent的回复添加到对话历史，用于后续多轮上下文
                    if not any(isinstance(msg, AIMessage) and msg.content == latest_message.content for msg in conversation_history):
                        conversation_history.append(AIMessage(content=latest_message.content))
                    # 输出Agent的回复
                    print(f"\n✅ Agent回答：{latest_message.content}")
                    print("\n" + "-" * 60)  # 分隔线优化体验

        except Exception as e:
            logger.error(f"Agent处理问题时出错：{str(e)}", exc_info=True)
            print(f"\n❌ 处理失败：{str(e)}")
            print("请重试或输入其他问题。\n" + "-" * 60)
            # 出错时移除当前的用户输入，避免污染对话历史
            conversation_history.pop()



if __name__ == "__main__":
    import asyncio
    # Python 3.14 推荐使用 asyncio.run
    try:
        asyncio.run(multi_round_chat())
    except Exception as e:
        print(f"运行中出现错误: {e}")