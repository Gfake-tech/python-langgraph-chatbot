"""
/**
 * @name LangGraph PDF Chatbot
 * @author Md. Samiur Rahman (Mukul)
 * @description LangGraph PDF Chatbot — AI-powered PDF Q&A System ~ Developed By Md. Samiur Rahman (Mukul)
 * @copyright ©2026 ― Md. Samiur Rahman (Mukul). All rights reserved.
 * @version v0.0.1
 *
 */
"""

"""
LangGraph PDF Chatbot - Graph Definition
========================================

KEY LANGGRAPH CONCEPTS:

1. STATE   – TypedDict shared by ALL nodes ("working memory")
2. NODE    – plain Python function: receives state → returns dict of updates
3. EDGE    – connection between nodes (normal or conditional)
4. ROUTER  – function that returns a STRING to pick the next node
5. GRAPH   – StateGraph assembled then .compile()d into a Runnable

Flow:
  START → [router] → generate_questions   ← PDF just uploaded
                   → answer_with_context  ← chat with PDF loaded
                   → web_search           ← web-search mode, no PDF
                   → chat                 ← general chat, no PDF
           → END
"""

import os
import threading

from dotenv import load_dotenv
from typing import TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_openai import AzureChatOpenAI
from tavily import TavilyClient

load_dotenv()


# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION  (all values read from .env)
# ─────────────────────────────────────────────────────────────────────────────

AZURE_ENDPOINT    = os.getenv("AZURE_OPENAI_ENDPOINT",    "")
AZURE_API_KEY     = os.getenv("AZURE_OPENAI_API_KEY",     "")
AZURE_DEPLOYMENT  = os.getenv("AZURE_OPENAI_DEPLOYMENT",  "gpt-5.2-chat")
AZURE_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2025-04-01-preview")
TAVILY_API_KEY    = os.getenv("TAVILY_API_KEY",           "")

_missing = [k for k, v in {
    "AZURE_OPENAI_ENDPOINT": AZURE_ENDPOINT,
    "AZURE_OPENAI_API_KEY":  AZURE_API_KEY,
}.items() if not v]
if _missing:
    raise EnvironmentError(
        f"Missing required environment variables: {_missing}\n"
        "Create a .env file (copy .env.example) and fill in your credentials."
    )


def get_llm() -> AzureChatOpenAI:
    return AzureChatOpenAI(
        azure_endpoint=AZURE_ENDPOINT,
        azure_deployment=AZURE_DEPLOYMENT,
        api_key=AZURE_API_KEY,
        api_version=AZURE_API_VERSION,
        max_tokens=2048,
    )


def get_tavily() -> TavilyClient:
    if not TAVILY_API_KEY:
        raise EnvironmentError(
            "TAVILY_API_KEY is not set in .env — required for web search."
        )
    return TavilyClient(api_key=TAVILY_API_KEY)


# ─────────────────────────────────────────────────────────────────────────────
# EXECUTION TRACE  (thread-safe — lets the GUI show which node ran)
# ─────────────────────────────────────────────────────────────────────────────

_trace = threading.local()


def reset_trace() -> None:
    _trace.nodes  = []
    _trace.reason = ""


def get_trace() -> dict:
    return {
        "nodes":  getattr(_trace, "nodes",  []),
        "reason": getattr(_trace, "reason", ""),
    }


def _record(node_name: str) -> None:
    if not hasattr(_trace, "nodes"):
        _trace.nodes = []
    _trace.nodes.append(node_name)


# ─────────────────────────────────────────────────────────────────────────────
# ① STATE
# ─────────────────────────────────────────────────────────────────────────────

class ChatState(TypedDict):
    messages          : Annotated[list, add_messages]  # auto-appended
    pdf_text          : str                             # extracted PDF text
    has_pdf           : bool                            # PDF loaded?
    action            : str                             # routing hint
    web_search_enabled: bool                            # Tavily toggle


# ─────────────────────────────────────────────────────────────────────────────
# ② ROUTER
# ─────────────────────────────────────────────────────────────────────────────

def route_action(state: ChatState) -> str:
    action             = state.get("action", "chat")
    has_pdf            = state.get("has_pdf", False)
    web_search_enabled = state.get("web_search_enabled", False)

    if action == "generate_questions":
        _trace.reason = "action='generate_questions' → PDF was just uploaded"
        return "generate_questions"

    if action == "chat" and has_pdf:
        _trace.reason = "action='chat' AND has_pdf=True → answer using document"
        return "answer_with_context"

    if action == "chat" and web_search_enabled and not has_pdf:
        _trace.reason = "web_search=True AND no PDF → searching the web with Tavily"
        return "web_search"

    _trace.reason = "action='chat', no PDF, no web search → general conversation"
    return "chat"


# ─────────────────────────────────────────────────────────────────────────────
# ③ NODES
# ─────────────────────────────────────────────────────────────────────────────

def generate_questions_node(state: ChatState) -> dict:
    """Summarises the uploaded PDF and proposes 6 questions."""
    _record("generate_questions")

    pdf_text = state.get("pdf_text", "")
    if not pdf_text:
        return {
            "messages": [AIMessage(content="No PDF content found. Please upload a PDF first.")],
            "action": "chat",
        }

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "You are an expert document analyst. Respond with:\n\n"
            "**Document Summary:**\n"
            "2–3 sentences summarising the main topics.\n\n"
            "**Questions you can ask me:**\n"
            "List exactly 6 numbered, insightful questions answerable from the document."
        )),
        HumanMessage(content=f"Document:\n\n{pdf_text[:10_000]}"),
    ])

    return {
        "messages": [AIMessage(content=response.content)],
        "action":   "chat",
    }


def answer_with_context_node(state: ChatState) -> dict:
    """Answers questions using the PDF text as context (simplified RAG)."""
    _record("answer_with_context")

    context = state.get("pdf_text", "")[:10_000]
    system_prompt = (
        "You are a helpful AI assistant that answers questions about a document.\n\n"
        "DOCUMENT:\n"
        "────────────────────────────────────────\n"
        f"{context}\n"
        "────────────────────────────────────────\n\n"
        "- Answer only from the document above.\n"
        "- Quote relevant passages when useful.\n"
        "- If the answer is not in the document, say so clearly."
    )

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        *state["messages"],
    ])

    return {"messages": [response], "action": "chat"}


def web_search_node(state: ChatState) -> dict:
    """
    NODE: web_search
    Uses Tavily to search the web, then passes the results to the LLM
    so it can answer the user's question with up-to-date information.

    Steps:
      1. Extract the latest user question from state.messages
      2. Call Tavily search API
      3. Format the top results as context
      4. Invoke the LLM with that context + conversation history
    """
    _record("web_search")

    # ── 1. Get the user's latest question ─────────────────────────────────────
    messages = state.get("messages", [])
    user_question = ""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            content = msg.content
            user_question = content if isinstance(content, str) else str(content)
            break

    if not user_question:
        return {
            "messages": [AIMessage(content="I couldn't find a question to search for.")],
            "action": "chat",
        }

    # ── 2. Tavily search ───────────────────────────────────────────────────────
    try:
        tavily  = get_tavily()
        results = tavily.search(
            query=user_question,
            max_results=5,
            search_depth="basic",   # "basic" is faster; use "advanced" for depth
            include_answer=True,    # Tavily's own AI-generated summary
        )
    except Exception as exc:
        return {
            "messages": [AIMessage(content=f"Web search failed: {exc}")],
            "action": "chat",
        }

    # ── 3. Format results ──────────────────────────────────────────────────────
    tavily_answer = results.get("answer", "")          # Tavily's own summary
    raw_results   = results.get("results", [])

    sources_text = "\n\n".join(
        f"**[{i+1}] {r.get('title', 'No title')}**\n"
        f"URL: {r.get('url', '')}\n"
        f"{r.get('content', '')[:400]}"
        for i, r in enumerate(raw_results)
    )

    search_context = ""
    if tavily_answer:
        search_context += f"**Tavily Quick Answer:**\n{tavily_answer}\n\n"
    if sources_text:
        search_context += f"**Source Articles:**\n{sources_text}"

    # ── 4. LLM synthesises a final answer ─────────────────────────────────────
    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "You are a helpful AI assistant with access to live web search results.\n\n"
            "WEB SEARCH RESULTS:\n"
            "─────────────────────────────────────\n"
            f"{search_context}\n"
            "─────────────────────────────────────\n\n"
            "Use the search results above to answer the user's question accurately. "
            "Always cite sources with [1], [2], etc. where relevant. "
            "If the results don't fully answer the question, say so."
        )),
        *messages,
    ])

    return {"messages": [response], "action": "chat"}


def chat_node(state: ChatState) -> dict:
    """General conversation — no PDF, no web search."""
    _record("chat")

    llm = get_llm()
    response = llm.invoke([
        SystemMessage(content=(
            "You are a helpful AI assistant. "
            "Tip: upload a PDF for document Q&A, or enable Web Search for live results!"
        )),
        *state["messages"],
    ])

    return {"messages": [response], "action": "chat"}


# ─────────────────────────────────────────────────────────────────────────────
# ④ BUILD THE GRAPH
# ─────────────────────────────────────────────────────────────────────────────

def create_chatbot_graph():
    graph = StateGraph(ChatState)

    graph.add_node("generate_questions",  generate_questions_node)
    graph.add_node("answer_with_context", answer_with_context_node)
    graph.add_node("web_search",          web_search_node)
    graph.add_node("chat",                chat_node)

    graph.add_conditional_edges(
        START,
        route_action,
        {
            "generate_questions":  "generate_questions",
            "answer_with_context": "answer_with_context",
            "web_search":          "web_search",
            "chat":                "chat",
        },
    )

    graph.add_edge("generate_questions",  END)
    graph.add_edge("answer_with_context", END)
    graph.add_edge("web_search",          END)
    graph.add_edge("chat",                END)

    return graph.compile()


chatbot_graph = create_chatbot_graph()
