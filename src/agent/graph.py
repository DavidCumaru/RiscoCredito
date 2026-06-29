"""LangGraph agent with tool-calling for credit risk analysis."""

from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent

from src.config import LLM_PROVIDER, OLLAMA_BASE_URL, OLLAMA_MODEL
from src.config import GROQ_API_KEY, GROQ_MODEL, GEMINI_API_KEY, GEMINI_MODEL
from src.agent.tools import ALL_TOOLS
from src.agent.rag import query_credit_policy

SYSTEM_PROMPT = """You are a senior credit risk analyst AI assistant. You help users assess loan applications, explain credit decisions, simulate what-if scenarios, and answer questions about the internal credit policy.

You have access to these tools:
- predict_credit_risk: Predict default probability for a loan applicant
- explain_credit_decision: Explain the key factors behind a prediction using SHAP
- simulate_scenario: Run what-if simulations by changing one variable
- query_credit_policy: Search the internal credit policy document

Guidelines:
- Always respond in Portuguese (Brazil).
- When a user describes an applicant, extract the parameters and call predict_credit_risk.
- If the user asks WHY a decision was made, use explain_credit_decision.
- For "what if" questions, use simulate_scenario.
- For policy questions (limits, rules, requirements), use query_credit_policy.
- Present results clearly with risk band, decision, and key factors.
- You may combine multiple tools in one response if needed.
- If information is missing, ask the user for it before calling a tool.
"""

TOOLS = ALL_TOOLS + [query_credit_policy]

_session_memory: dict[str, list] = {}


def _get_llm():
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(api_key=GROQ_API_KEY, model_name=GROQ_MODEL, temperature=0)
    elif LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            google_api_key=GEMINI_API_KEY, model=GEMINI_MODEL, temperature=0,
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL, temperature=0)


def get_agent():
    llm = _get_llm()
    return create_react_agent(llm, TOOLS, prompt=SystemMessage(content=SYSTEM_PROMPT))


def chat(session_id: str, user_message: str) -> str:
    """Run the agent with session memory."""
    agent = get_agent()

    if session_id not in _session_memory:
        _session_memory[session_id] = []

    history = _session_memory[session_id]
    history.append(HumanMessage(content=user_message))

    result = agent.invoke({"messages": history})

    response_messages = result["messages"]
    _session_memory[session_id] = response_messages

    ai_messages = [m for m in response_messages if hasattr(m, "content") and m.type == "ai" and m.content]
    if ai_messages:
        return ai_messages[-1].content

    return "Desculpe, não consegui processar sua solicitação."


def clear_session(session_id: str) -> None:
    _session_memory.pop(session_id, None)
