import os
import json
import requests
from dotenv import load_dotenv
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
# ========== 核心修改1：适配1.2.0版本的导入路径 ==========
from langchain_community.chat_models.tongyi import ChatTongyi
from langgraph.checkpoint.memory import MemorySaver  # 记忆组件新路径
from langchain.tools import tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # 核心组件新路径
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool  # 1.x版本工具注册新方式

# 加载环境变量
load_dotenv()

# ====================== 1. 定义Agent状态（核心：统一管理所有上下文） ======================
class AgentState(TypedDict):
    user_input: str                # 用户当前输入
    chat_history: List[Any]        # 对话历史（存储HumanMessage/AIMessage）
    tool_name: str                 # 要调用的工具名称
    tool_input: Dict[str, str]     # 工具输入参数
    tool_result: str               # 工具调用结果
    final_answer: str              # 最终回答

# 初始化默认状态（避免空值报错）
def get_initial_state() -> AgentState:
    return {
        "user_input": "",
        "chat_history": [],
        "tool_name": "",
        "tool_input": {},
        "tool_result": "",
        "final_answer": ""
    }


# ====================== 2. 定义工具（适配1.2.0版本） ======================
# 工具1：天气查询
@tool
def weather_query(city: str) -> str:
    """
    查询指定城市的实时天气
    :param city: 城市名称（如北京、上海）
    :return: 天气信息字符串
    """
    try:
        url = f"http://wthrcdn.etouch.cn/weather_mini?city={city}"
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if data["desc"] != "success":
            return f"查询失败：{data['desc']}"
        
        weather_info = data["data"]["forecast"][0]
        return (
            f"{city}今日天气：{weather_info['type']}，"
            f"气温{weather_info['low']}~{weather_info['high']}，"
            f"风向{weather_info['fengxiang']}，风力{weather_info['fengli']}"
        )
    except Exception as e:
        return f"天气查询异常：{str(e)}"

# 工具2：计算器
@tool
def calculator(expression: str) -> str:
    """
    执行简单数学计算（支持加减乘除、括号）
    :param expression: 数学表达式（如1+2*3、(10-5)/2）
    :return: 计算结果
    """
    try:
        # 安全执行表达式（避免恶意代码）
        result = eval(expression, {"__builtins__": None}, {})
        return f"计算结果：{expression} = {result}"
    except Exception as e:
        return f"计算失败：{str(e)}"

# ========== 核心修改2：1.x版本工具映射方式微调 ==========
tools = {
    "weather_query": weather_query,
    "calculator": calculator
}

# ====================== 3. 定义LangGraph节点 ======================
# 初始化LLM
llm = ChatTongyi(
    model="qwen-turbo",  # 千问模型版本（qwen-turbo/qwen-plus/qwen-max）
    temperature=0,  # 输出稳定，便于调试
    dashscope_api_key=os.getenv("DASHSCOPE_API_KEY"),  # 阿里云API Key
    verbose=True
)

# 节点1：思考决策（返回字典状态）
def think_node(state: AgentState) -> AgentState:
    user_input = state["user_input"].strip()
    # 空输入直接返回“无工具”，避免LLM无意义调用
    if not user_input:
        return {**state, "tool_name": "", "tool_input": {}}
    
    # 优化提示词：明确要求空工具时返回空字符串，且禁止冗余回复
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""
你是决策助手，仅判断是否调用工具，规则：
1. 仅当用户输入是「查天气」（含城市名）或「数学计算」（含表达式）时，返回对应工具信息；
2. 其他情况（如问候、闲聊、空输入），返回{"tool_name":"","tool_input":{}}；
3. 必须返回标准JSON，无任何多余文字、注释、换行。
示例1（查天气）：{"tool_name":"weather_query","tool_input":{"city":"北京"}}
示例2（计算）：{"tool_name":"calculator","tool_input":{"expression":"1+1"}}
示例3（闲聊）：{"tool_name":"","tool_input":{}}
"""),
        HumanMessage(content=user_input)
    ])

    # 调用LLM并增加容错
    try:
        llm_output = prompt | llm
        decision = llm_output.invoke({"chat_history": state["chat_history"]})
        decision_dict = json.loads(decision.content.strip())
    except Exception as e:
        print(f"[调试] LLM决策解析失败：{e}，使用默认值")
        decision_dict = {"tool_name": "", "tool_input": {}}
    
    # 强制校验tool_name类型
    tool_name = decision_dict.get("tool_name", "").strip()
    tool_input = decision_dict.get("tool_input", {})
    
    return {**state, "tool_name": tool_name, "tool_input": tool_input}

# 节点2：工具执行（返回字典状态）
def tool_node(state: AgentState) -> AgentState:
    tool_name = state["tool_name"]
    tool_input = state["tool_input"]
    
    # 执行工具
    if tool_name not in tools:
        tool_result = f"未知工具：{tool_name}"
    else:
        try:
            tool_result = tools[tool_name].invoke(tool_input)
        except Exception as e:
            tool_result = f"工具执行出错：{str(e)}"

    # 返回更新后的状态字典
    return {
        **state,
        "tool_result": tool_result
    }


# 节点3：生成最终回答（返回字典状态）
def answer_node(state: AgentState) -> AgentState:
    user_input = state["user_input"].strip()
    tool_result = state["tool_result"]
    chat_history = state["chat_history"]

    # # 🔥 调试：打印节点输入的chat_history（确认入参是否为空）
    # print(f"\n===== [answer_node 输入] =====")
    # print(f"输入chat_history条数：{len(chat_history)}")
    # print(f"输入chat_history内容：{[msg.content for msg in chat_history] if chat_history else '空'}")

    # 空输入时简化回复
    if not user_input:
        final_answer = "请问你有什么具体问题需要帮助？比如查询某个城市的天气，或执行数学计算。"
    else:
        # 优化提示词：禁止重复、冗余回复
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""
你是一个有记忆的助手，要考虑对话历史（chat_history）回答用户问题，回答规则：
1. 简洁友好，不重复、不啰嗦；
2. 有工具结果则基于工具结果回答，无则直接回应用户（如问候）；
3. 仅回复用户当前问题，不主动追问、不额外输出无关内容；
4. 如果是工具调用失败，要回答失败原因
5.如果用户问“前面问了什么”“上一个问题是什么”，必须从chat_history中提取并回答
"""),
            MessagesPlaceholder(variable_name="chat_history"),
            HumanMessage(content=user_input),
            SystemMessage(content=f"工具结果：{tool_result if tool_result else '无'}")
        ])
        final_answer = (prompt | llm).invoke({
            "chat_history": chat_history,
            "user_input": user_input,
            "tool_result": tool_result
        }).content
    
    # 更新对话历史（避免重复添加）
    new_chat_history = chat_history.copy()
    if user_input:  # 空输入不添加到历史
        new_chat_history.append(HumanMessage(content=user_input))
        new_chat_history.append(AIMessage(content=final_answer))
    # # 🔥 调试：打印节点输出的chat_history（确认是否生成）
    # print(f"\n===== [answer_node 输出] =====")
    # print(f"输出chat_history条数：{len(new_chat_history)}")
    # print(f"输出chat_history内容：{[msg.content for msg in new_chat_history]}")

    return {**state, "final_answer": final_answer, "chat_history": new_chat_history}

# ====================== 4. 构建LangGraph流程 ======================
def build_agent_graph():
    # 初始化StateGraph（传入TypedDict类型）
    graph = StateGraph(AgentState)
    
    # 添加节点
    graph.add_node("think", think_node)
    graph.add_node("tool", tool_node)
    graph.add_node("answer", answer_node)

    # 2. 修正条件边：明确指定分支目标（避免LangGraph解析失败）
    def should_call_tool(state: AgentState) -> str:
        # 增加日志，调试条件边返回值
        print(f"[调试] think节点决策：tool_name={state['tool_name']}")
        # 严格判断：有工具名则走tool节点，否则走answer节点
        return "tool" if state["tool_name"].strip() else "answer"

    # 3. 重新定义所有边（确保顺序/关联正确）
    graph.set_entry_point("think")  # 入口是think
    # 条件边：think → tool/answer（核心！之前可能未正确添加）
    graph.add_conditional_edges(
        source="think",          # 源节点
        path=should_call_tool,   # 分支函数
        path_map={               # 显式指定分支目标（兜底，避免解析失败）
            "tool": "tool",
            "answer": "answer"
        }
    )
    # 工具节点执行完 → answer节点
    graph.add_edge("tool", "answer")
    # answer节点执行完 → 结束
    graph.add_edge("answer", END)

    # 接入MemorySaver（核心：Checkpoint适配TypedDict）
    memory_saver = MemorySaver()
    return graph.compile(checkpointer=memory_saver)

# 新增：打印Checkpoint内容的函数（核心调试）
def debug_checkpoint(agent_graph, thread_id):
    """鲁棒的Checkpoint读取函数，处理空值+结构层级"""
    print("\n===== [记忆调试] Checkpoint 详情 =====")
    try:
        # 1. 读取Checkpoint（允许返回None）
        checkpoint = agent_graph.checkpointer.get(config={"configurable": {"thread_id": thread_id}})
        
        # 2. 处理空Checkpoint（首次对话/无状态）
        if not checkpoint:
            print("❌ 未找到该会话的Checkpoint（首次对话/无状态）")
            return
       
        # 3. 正确解析Checkpoint结构（关键：先取checkpoint层级，再取values）
        # LangGraph Checkpoint 完整结构：
#         {
#            "v": 4,
#           "ts": "2025-12-23T15:38:42.692587+00:00",
#            "id": "1f0e0157-5fcd-6ba3-8002-2682dbb5c2fd",
#           "think": {
#                "branch:to:think": "00000000000000000000000000000002.0.5896163833832954"
#           },
#           "answer": {
#               "branch:to:answer": "00000000000000000000000000000003.0.4838919578982862"
#           }
#           },
#   "channel_values": {
#     "user_input": "你好",
#     "chat_history": [
#       {
#         "type": "human",
#         "content": "你好"
#       },
#       {
#         "type": "ai",
#         "content": "你好！有什么可以帮你的吗？"
#       }
#     ],
#     "tool_name": "",
#     "tool_input": {},
#     "tool_result": "",
#     "final_answer": "你好！有什么可以帮你的吗？"
#   }
# }
    
        chat_history = checkpoint.get("channel_values", {}).get("chat_history", [])
        
        # 4. 打印记忆详情
        print(f"✅ 会话ID：{thread_id}")
        print(f"✅ Checkpoint是否存在：是")
        print(f"✅ 对话历史条数：{len(chat_history)}")
        
        if len(chat_history) == 0:
            print("⚠️ 对话历史为空（已生成Checkpoint，但无历史消息）")
        else:
            print("✅ 对话历史内容：")
            for idx, msg in enumerate(chat_history):
                role = "用户" if isinstance(msg, HumanMessage) else "Agent"
                print(f"  {idx+1}. {role}：{msg.content}")
                
    except Exception as e:
        print(f"❌ 读取Checkpoint失败：{str(e)}")
        # 打印原始Checkpoint结构（方便排查）
        print(f"❌ 原始Checkpoint结构：{checkpoint if 'checkpoint' in locals() else 'None'}")

# ====================== 5. 运行Agent ======================
if __name__ == "__main__":
    # 构建Agent（带MemorySaver）
    agent_graph = build_agent_graph()

    try:
    # 1. 临时编译无checkpointer的图（避免MemorySaver干扰）
        temp_graph = agent_graph.get_graph()
    # 2. 优先尝试Ascii绘图，失败则生成Mermaid文本（更稳定）
        print("===== 图结构（ASCII）=====")
        print(temp_graph.draw_ascii())
    except Exception as e:
        print(f"ASCII绘图失败：{str(e)}")
    # 备选：生成Mermaid流程图（可复制到https://mermaid.live/可视化）
    # try:
    #     print("\n===== 图结构（Mermaid）=====")
    #     mermaid_code = temp_graph.draw_mermaid()
    #     print(mermaid_code)
    # except Exception as e2:
    #     print(f"Mermaid绘图也失败：{str(e2)}")
    # 会话ID（区分不同用户）
    thread_id = "user_001"
    # 初始化状态（字典格式）
    
    current_state = get_initial_state()
    print("===== 多功能助手Agent（LangGraph MemorySaver会话记忆版） =====")
    print("支持：天气查询、数学计算、持久化记忆 | 输入'退出'结束\n")

    while True:
        user_input = input("你：")
        if user_input.strip() == "退出":
            print("Agent：再见！")
            break
        
        # 更新用户输入（字典状态）
        current_state["user_input"] = user_input
        
        # 调用Agent（核心：config格式适配 + 字典输入）
        final_state = agent_graph.invoke(
            input=current_state,
            config={"configurable": {"thread_id": thread_id}}
        )
        # 🔥 核心修复：将final_state的chat_history同步回current_state
        # （Checkpoint保存的是final_state，必须同步才能在下一轮传递）
        current_state["chat_history"] = final_state["chat_history"]
        # 输出结果
        print(f"Agent：{final_state['final_answer']}\n")
         # 🔥 关键：每轮对话后打印Checkpoint，验证记忆是否保存
        debug_checkpoint(agent_graph, thread_id)
        # 重置工具相关状态（保留chat_history）
        current_state["tool_name"] = ""
        current_state["tool_input"] = {}
        current_state["tool_result"] = ""
        current_state["final_answer"] = ""
        # chat_history由MemorySaver自动持久化，无需手动维护

