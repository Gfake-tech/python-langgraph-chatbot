"""
LangGraph PDF Chatbot — Educational GUI
=======================================
Run:  python app.py  →  http://localhost:7860

Educational features:
  • LangGraph Flow Visualizer   — shows nodes, edges, and HIGHLIGHTS the active node
  • Execution Trace panel       — shows START → router → node → END after each message
  • State Inspector panel       — shows the full ChatState as JSON after each step
  • "Learn LangGraph" tab       — concept explanations with code from THIS project
"""

import os, json, textwrap
import gradio as gr
from pypdf import PdfReader
from langchain_core.messages import HumanMessage, AIMessage

from graph import chatbot_graph, ChatState, reset_trace, get_trace


# ═══════════════════════════════════════════════════════════════════════════════
# PDF HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def extract_pdf_text(filepath: str) -> str:
    try:
        reader = PdfReader(filepath)
        pages  = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"[Page {i + 1}]\n{text.strip()}")
        return "\n\n".join(pages)
    except Exception as exc:
        print(f"PDF error: {exc}")
        return ""


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO ↔ LANGCHAIN BRIDGE
# ═══════════════════════════════════════════════════════════════════════════════

def _extract_text(content) -> str:
    """
    Safely extract plain text from a Gradio message content field.
    Gradio 6 can return either a plain string OR a list of content blocks
    (e.g. [{"type": "text", "text": "..."}, ...]) for multimodal messages.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        # Concatenate all text blocks
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                parts.append(block.get("text") or block.get("content") or "")
        return " ".join(parts).strip()
    return str(content).strip()


def history_to_langchain(history: list) -> list:
    """Convert Gradio message-dict history into LangChain message objects."""
    msgs = []
    for m in history:
        role    = m.get("role", "")
        content = _extract_text(m.get("content"))
        if not content:
            continue
        if role == "user" and not content.startswith("[PDF:"):
            msgs.append(HumanMessage(content=content))
        elif role == "assistant":
            msgs.append(AIMessage(content=content))
    return msgs

def _bot(text): return {"role": "assistant", "content": text}
def _usr(text): return {"role": "user",      "content": text}


# ═══════════════════════════════════════════════════════════════════════════════
# LANGGRAPH FLOW VISUALIZER  (HTML)
#
# Renders a CSS flowchart of the graph.
# Pass `active_node` to highlight which node was just executed.
# ═══════════════════════════════════════════════════════════════════════════════

_NODE_LABELS = {
    "generate_questions":  "generate_questions_node",
    "answer_with_context": "answer_with_context_node",
    "web_search":          "web_search_node",
    "chat":                "chat_node",
}

_NODE_DESCRIPTIONS = {
    "generate_questions":  "Analyses the PDF and produces a summary + 6 questions",
    "answer_with_context": "Answers your question using the PDF as context (RAG)",
    "web_search":          "Searches the web via Tavily and synthesises an answer",
    "chat":                "General conversation — no PDF, no web search",
}

def build_graph_html(active_node: str = None, route_reason: str = "") -> str:
    """
    Compact VERTICAL flow diagram — designed to fit inside a narrow right-panel card
    without any horizontal overflow.

    Layout:
        ● START
          ↓
       route_action()
          ↓
       [node row]  ← all three nodes stacked vertically
       [node row]
       [node row]
          ↓
        ● END
        ── router reason ──
    """

    _CONDITIONS = {
        "generate_questions":  'action = "generate_questions"',
        "answer_with_context": 'action = "chat"  +  has_pdf = True',
        "web_search":          'action = "chat"  +  web_search = True',
        "chat":                'action = "chat"  +  no PDF  +  no web search',
    }

    def node_row(key: str) -> str:
        label    = _NODE_LABELS[key]
        desc     = _NODE_DESCRIPTIONS[key]
        cond     = _CONDITIONS[key]
        is_active = key == active_node

        if is_active:
            bg      = "#4f46e5"
            fg      = "#fff"
            border  = "2px solid #3730a3"
            shadow  = "box-shadow:0 0 0 3px #c7d2fe;"
            dot     = "●"
            dot_clr = "#a5b4fc"
            badge   = ('<span style="background:#10b981;color:#fff;font-size:9px;'
                       'padding:1px 7px;border-radius:99px;margin-left:6px;'
                       'font-weight:700;vertical-align:middle;">RAN</span>')
        else:
            bg      = "#f8fafc"
            fg      = "#94a3b8"
            border  = "1.5px solid #e2e8f0"
            shadow  = ""
            dot     = "○"
            dot_clr = "#cbd5e1"
            badge   = ""

        return f"""
        <div style="display:flex;align-items:flex-start;gap:10px;
                    padding:9px 11px;border-radius:9px;margin-bottom:6px;
                    background:{bg};color:{fg};border:{border};{shadow}">
          <span style="font-size:13px;color:{dot_clr};flex-shrink:0;margin-top:1px;">{dot}</span>
          <div style="min-width:0;flex:1;">
            <div style="font-size:11px;font-weight:700;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
              {label}{badge}
            </div>
            <div style="font-size:10px;opacity:.75;margin-top:2px;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
              {desc}
            </div>
            <div style="font-size:9px;opacity:.6;margin-top:3px;font-style:italic;
                        white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
              when: {cond}
            </div>
          </div>
        </div>"""

    arrow = ('<div style="text-align:center;color:#94a3b8;'
             'font-size:18px;line-height:1;margin:3px 0;">↓</div>')

    def pill(color, text):
        return (f'<div style="text-align:center;margin:4px 0;">'
                f'<span style="background:{color};color:#fff;padding:4px 16px;'
                f'border-radius:99px;font-size:11px;font-weight:700;">{text}</span></div>')

    reason_html = ""
    if route_reason:
        reason_html = f"""
        <div style="margin-top:10px;padding:8px 10px;
                    background:#fef9c3;border-left:3px solid #eab308;
                    border-radius:6px;font-size:10px;color:#713f12;line-height:1.5;">
          <b>Why:</b> {route_reason}
        </div>"""

    return f"""
    <div style="font-family:'Inter',system-ui,sans-serif;
                padding:4px 2px;box-sizing:border-box;width:100%;overflow:hidden;">
      {pill('#10b981', '● START')}
      {arrow}
      <div style="text-align:center;background:#fff7ed;border:1.5px solid #fed7aa;
                  border-radius:8px;padding:7px 10px;margin:0 0 4px;">
        <div style="font-size:11px;font-weight:700;color:#c2410c;">route_action()</div>
        <div style="font-size:9px;color:#9a3412;margin-top:2px;">
          reads state.action + state.has_pdf
        </div>
      </div>
      {arrow}
      <div>
        {node_row('generate_questions')}
        {node_row('answer_with_context')}
        {node_row('web_search')}
        {node_row('chat')}
      </div>
      {arrow}
      {pill('#ef4444', '● END')}
      {reason_html}
    </div>"""


# ═══════════════════════════════════════════════════════════════════════════════
# STATE INSPECTOR  (JSON view)
# ═══════════════════════════════════════════════════════════════════════════════

def build_state_html(state_snapshot: dict) -> str:
    """Render the ChatState as colour-coded HTML."""
    display = {}
    for k, v in state_snapshot.items():
        if k == "messages":
            display["messages"] = [
                {"role": type(m).__name__, "content": (m.content[:120] + "…") if len(m.content) > 120 else m.content}
                for m in v
            ]
        elif k == "pdf_text":
            chars = len(v) if v else 0
            display["pdf_text"] = f"[{chars:,} characters]" if chars else "(empty)"
        else:
            display[k] = v

    pretty = json.dumps(display, indent=2, ensure_ascii=False)

    # Syntax-highlight: keys in blue, strings in green, booleans in orange
    import re
    pretty = re.sub(r'"([^"]+)":', r'<span style="color:#2563eb;">"<b>\1</b>"</span>:', pretty)
    pretty = re.sub(r': "([^"]*)"', r': <span style="color:#059669;">"<i>\1</i>"</span>', pretty)
    pretty = re.sub(r': (true|false)', r': <span style="color:#d97706;">\1</span>', pretty)

    return f"""
    <div style="background:#0f172a;color:#e2e8f0;border-radius:10px;
                padding:14px;font-family:'Courier New',monospace;font-size:12px;
                overflow:auto;max-height:280px;line-height:1.6;">
      <div style="color:#64748b;margin-bottom:8px;font-size:11px;">
        ChatState  ·  {len(state_snapshot.get('messages', []))} messages
        {'  ·  PDF loaded' if state_snapshot.get('has_pdf') else ''}
      </div>
      <pre style="margin:0;white-space:pre-wrap;">{pretty}</pre>
    </div>"""


# ═══════════════════════════════════════════════════════════════════════════════
# EXECUTION TRACE TEXT
# ═══════════════════════════════════════════════════════════════════════════════

def build_trace_html(nodes: list[str], reason: str) -> str:
    if not nodes:
        return "<p style='color:#94a3b8;font-size:13px;'>No trace yet — send a message!</p>"

    node_chips = "".join(
        f'<span style="background:#4f46e5;color:#fff;padding:4px 10px;'
        f'border-radius:6px;font-size:12px;font-weight:600;margin:0 2px;">{n}</span>'
        for n in nodes
    )

    return f"""
    <div style="font-family:'Courier New',monospace;font-size:13px;line-height:2;">
      <span style="background:#10b981;color:#fff;padding:4px 10px;border-radius:6px;font-size:12px;">START</span>
      <span style="color:#94a3b8;margin:0 6px;">→</span>
      <span style="background:#f59e0b;color:#fff;padding:4px 10px;border-radius:6px;font-size:12px;">route_action()</span>
      <span style="color:#94a3b8;margin:0 6px;">→</span>
      {node_chips}
      <span style="color:#94a3b8;margin:0 6px;">→</span>
      <span style="background:#ef4444;color:#fff;padding:4px 10px;border-radius:6px;font-size:12px;">END</span>
      <div style="margin-top:8px;padding:8px 12px;background:#fef9c3;border-left:3px solid #eab308;
                  border-radius:4px;font-size:12px;color:#713f12;">
        <b>Why?</b> {reason or "—"}
      </div>
    </div>"""


# ═══════════════════════════════════════════════════════════════════════════════
# LEARN LANGGRAPH TAB  (static educational content)
# ═══════════════════════════════════════════════════════════════════════════════

LEARN_HTML = """
<style>
  .lg-card { background:#f8fafc; border:1px solid #e2e8f0; border-radius:12px;
             padding:20px; margin-bottom:16px; }
  .lg-card h3 { margin:0 0 8px; color:#1e293b; font-size:15px; }
  .lg-card p  { margin:4px 0; color:#475569; font-size:13px; }
  .lg-badge   { display:inline-block; padding:2px 10px; border-radius:99px;
                font-size:11px; font-weight:700; margin-bottom:8px; }
  .lg-code    { background:#0f172a; color:#e2e8f0; border-radius:8px; padding:14px;
                font-family:'Courier New',monospace; font-size:12px;
                white-space:pre; overflow:auto; margin-top:10px; line-height:1.6; }
  .concept    { color:#4f46e5; }
  .kw         { color:#f472b6; }
  .fn         { color:#34d399; }
  .str        { color:#fb923c; }
  .cm         { color:#64748b; }
</style>

<h2 style="color:#1e293b;margin-bottom:4px;">🧠 LangGraph — Core Concepts</h2>
<p style="color:#64748b;margin-bottom:20px;font-size:13px;">
  5 things you need to understand. Every line of <code>graph.py</code> maps to one of these.
</p>

<!-- CONCEPT 1 -->
<div class="lg-card">
  <span class="lg-badge" style="background:#dbeafe;color:#1d4ed8;">① STATE</span>
  <h3>The shared memory that flows through every node</h3>
  <p>State is a <b>TypedDict</b> — a typed Python dict. Every node reads from it and returns
     updates to it. You never mutate state directly; nodes return a dict of changes.</p>
  <div class="lg-code"><span class="cm"># from graph.py — our State definition</span>
<span class="kw">class</span> <span class="concept">ChatState</span>(TypedDict):
    messages : Annotated[list, add_messages]  <span class="cm"># auto-appends new msgs</span>
    pdf_text : str                             <span class="cm"># extracted PDF content</span>
    has_pdf  : bool                            <span class="cm"># True once PDF is loaded</span>
    action   : str                             <span class="cm"># routing hint for next step</span></div>
  <p style="margin-top:10px;"><b>💡 add_messages:</b> Instead of overwriting the list, this reducer
     <em>appends</em> new messages — giving multi-turn memory for free.</p>
</div>

<!-- CONCEPT 2 -->
<div class="lg-card">
  <span class="lg-badge" style="background:#dcfce7;color:#16a34a;">② NODE</span>
  <h3>A plain Python function that does one job</h3>
  <p>Nodes receive the full state and return a <b>dict of only the fields they changed</b>.
     Fields you don't return keep their old values.</p>
  <div class="lg-code"><span class="cm"># from graph.py — a node that answers using PDF context</span>
<span class="kw">def</span> <span class="fn">answer_with_context_node</span>(state: <span class="concept">ChatState</span>) -> dict:
    llm      = get_llm()
    response = llm.invoke([
        SystemMessage(content=f<span class="str">"Document: {state['pdf_text'][:10000]}"</span>),
        *state[<span class="str">"messages"</span>],   <span class="cm"># full history</span>
    ])
    <span class="kw">return</span> {
        <span class="str">"messages"</span>: [response],  <span class="cm"># ← only return what changed</span>
        <span class="str">"action"</span>:   <span class="str">"chat"</span>,
    }</div>
</div>

<!-- CONCEPT 3 -->
<div class="lg-card">
  <span class="lg-badge" style="background:#fef3c7;color:#d97706;">③ ROUTER</span>
  <h3>A function that decides which node to run next</h3>
  <p>The router reads the state and returns a <b>string</b>.
     LangGraph maps that string to a node name.</p>
  <div class="lg-code"><span class="cm"># from graph.py — our conditional router</span>
<span class="kw">def</span> <span class="fn">route_action</span>(state: <span class="concept">ChatState</span>) -> str:
    action  = state.get(<span class="str">"action"</span>, <span class="str">"chat"</span>)
    has_pdf = state.get(<span class="str">"has_pdf"</span>, False)

    <span class="kw">if</span> action == <span class="str">"generate_questions"</span>:
        <span class="kw">return</span> <span class="str">"generate_questions"</span>   <span class="cm"># → goes to that node</span>

    <span class="kw">if</span> action == <span class="str">"chat"</span> <span class="kw">and</span> has_pdf:
        <span class="kw">return</span> <span class="str">"answer_with_context"</span>    <span class="cm"># → uses PDF context</span>

    <span class="kw">return</span> <span class="str">"chat"</span>                      <span class="cm"># → general conversation</span></div>
</div>

<!-- CONCEPT 4 -->
<div class="lg-card">
  <span class="lg-badge" style="background:#fce7f3;color:#be185d;">④ EDGE</span>
  <h3>Connections between nodes (normal or conditional)</h3>
  <p><b>Normal edge</b> — always goes to the same next node.<br>
     <b>Conditional edge</b> — calls a router function to decide where to go.</p>
  <div class="lg-code"><span class="cm"># from graph.py — adding edges</span>
<span class="cm"># Conditional: START uses route_action() to pick a node</span>
graph.add_conditional_edges(
    START,
    <span class="fn">route_action</span>,              <span class="cm"># router function</span>
    {                              <span class="cm"># mapping: string → node name</span>
        <span class="str">"generate_questions"</span>:  <span class="str">"generate_questions"</span>,
        <span class="str">"answer_with_context"</span>: <span class="str">"answer_with_context"</span>,
        <span class="str">"chat"</span>:                <span class="str">"chat"</span>,
    },
)

<span class="cm"># Normal: every node leads straight to END</span>
graph.add_edge(<span class="str">"generate_questions"</span>,  END)
graph.add_edge(<span class="str">"answer_with_context"</span>, END)
graph.add_edge(<span class="str">"chat"</span>,                END)</div>
</div>

<!-- CONCEPT 5 -->
<div class="lg-card">
  <span class="lg-badge" style="background:#ede9fe;color:#7c3aed;">⑤ GRAPH</span>
  <h3>Assembling and compiling the workflow</h3>
  <p>Create a <b>StateGraph</b>, register nodes, add edges, then call <b>.compile()</b>
     to get a runnable object you call with <code>.invoke(state)</code>.</p>
  <div class="lg-code"><span class="cm"># from graph.py — the full assembly</span>
graph = <span class="concept">StateGraph</span>(<span class="concept">ChatState</span>)          <span class="cm"># bind to our State type</span>

graph.add_node(<span class="str">"generate_questions"</span>,  generate_questions_node)
graph.add_node(<span class="str">"answer_with_context"</span>, answer_with_context_node)
graph.add_node(<span class="str">"chat"</span>,                chat_node)

graph.add_conditional_edges(START, route_action, {...})
graph.add_edge(<span class="str">"generate_questions"</span>, END)
<span class="cm"># ... rest of edges ...</span>

chatbot_graph = graph.compile()    <span class="cm"># → Runnable object</span>

<span class="cm"># Usage: pass initial state, get final state back</span>
result = chatbot_graph.invoke({
    <span class="str">"messages"</span>: [HumanMessage(content=<span class="str">"Hello"</span>)],
    <span class="str">"has_pdf"</span>:  False,
    <span class="str">"action"</span>:   <span class="str">"chat"</span>,
})</div>
</div>

<div style="background:#f0fdf4;border:1px solid #86efac;border-radius:10px;padding:16px;margin-top:8px;">
  <b style="color:#166534;">✅ Summary — LangGraph in one sentence:</b><br>
  <p style="color:#166534;margin:6px 0 0;">
    Define a <b>State</b> (shared dict), write <b>Nodes</b> (functions that update it),
    connect them with <b>Edges</b> (optionally routing through a <b>Router</b>),
    compile into a <b>Graph</b>, and call <code>.invoke()</code>.
  </p>
</div>
"""


# ═══════════════════════════════════════════════════════════════════════════════
# CORE EVENT HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

def invoke_graph(state_dict: dict) -> tuple[dict, dict]:
    """
    Invoke the LangGraph and return (result_messages_last, trace_dict).
    Uses reset_trace() + get_trace() from graph.py to capture which node ran.
    """
    reset_trace()
    result = chatbot_graph.invoke(state_dict)
    trace  = get_trace()
    return result, trace


def handle_chat(user_msg: str, history: list, pdf_state: dict, web_search: bool = False):
    """
    User sent a chat message.
    Returns: (updated_history, pdf_state, clear_input,
               graph_html, trace_html, state_html)
    """
    user_msg = (user_msg or "").strip()
    if not user_msg:
        return history, pdf_state, "", gr.update(), gr.update(), gr.update()

    prior_msgs = history_to_langchain(history)
    prior_msgs.append(HumanMessage(content=user_msg))

    state: ChatState = {
        "messages":           prior_msgs,
        "pdf_text":           pdf_state.get("pdf_text", ""),
        "has_pdf":            pdf_state.get("has_pdf", False),
        "action":             "chat",
        "web_search_enabled": bool(web_search),
    }

    try:
        result, trace = invoke_graph(state)
        ai_reply = result["messages"][-1].content
    except Exception as exc:
        ai_reply = f"**Error:** {exc}\n\nCheck your API key / deployment name in `graph.py`."
        trace    = {"nodes": [], "reason": str(exc)}

    active_node = trace["nodes"][0] if trace["nodes"] else None
    history = history + [_usr(user_msg), _bot(ai_reply)]

    # Rebuild display state (approximation — messages from langchain objs)
    display_state = {**state, "messages": prior_msgs + [AIMessage(content=ai_reply)]}

    return (
        history,
        pdf_state,
        "",                                                          # clear input
        build_graph_html(active_node, trace["reason"]),              # graph viz
        build_trace_html(trace["nodes"], trace["reason"]),           # trace
        build_state_html(display_state),                             # state JSON
    )


def handle_pdf_upload(filepath: str | None, history: list, pdf_state: dict):
    if filepath is None:
        return history, pdf_state, "No file received.", gr.update(), gr.update(), gr.update()

    filename = os.path.basename(filepath)
    pdf_text = extract_pdf_text(filepath)

    if not pdf_text:
        history = history + [
            _usr(f"[PDF: {filename}]"),
            _bot("Could not extract text. Please use a text-based (not scanned) PDF."),
        ]
        return history, pdf_state, "Extraction failed.", gr.update(), gr.update(), gr.update()

    new_pdf_state = {"pdf_text": pdf_text, "has_pdf": True}

    state: ChatState = {
        "messages":           [HumanMessage(content="Analyse this document.")],
        "pdf_text":           pdf_text,
        "has_pdf":            True,
        "action":             "generate_questions",
        "web_search_enabled": False,
    }

    result = None
    try:
        result, trace = invoke_graph(state)
        analysis = result["messages"][-1].content
    except Exception as exc:
        analysis = f"**Error:** {exc}"
        trace    = {"nodes": ["generate_questions"], "reason": "PDF just uploaded → generate_questions"}

    word_count = len(pdf_text.split())
    header     = f"**PDF loaded:** `{filename}` — {word_count:,} words\n\n"
    history    = history + [_usr(f"[PDF: {filename}]"), _bot(header + analysis)]

    active_node   = trace["nodes"][0] if trace["nodes"] else "generate_questions"
    display_state = {**state, "messages": result["messages"] if result else state["messages"]}

    return (
        history,
        new_pdf_state,
        f"PDF loaded: {filename} ({word_count:,} words)",
        build_graph_html(active_node, trace["reason"]),
        build_trace_html(trace["nodes"], trace["reason"]),
        build_state_html(display_state),
    )


def handle_remove_pdf(history: list):
    empty = {"pdf_text": "", "has_pdf": False}
    history = history + [_bot("PDF removed — back to general chat mode.")]
    blank_state = {"messages": [], "pdf_text": "", "has_pdf": False, "action": "chat"}
    return (
        history, empty, "No PDF loaded",
        build_graph_html(),
        build_trace_html([], ""),
        build_state_html(blank_state),
    )


def handle_clear():
    welcome_state = {"messages": [], "pdf_text": "", "has_pdf": False, "action": "chat"}
    welcome = [_bot("Chat cleared! Upload a PDF or just start chatting.")]
    return (
        welcome, {"pdf_text": "", "has_pdf": False}, "No PDF loaded",
        build_graph_html(),
        build_trace_html([], ""),
        build_state_html(welcome_state),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GRADIO UI
# Two-column layout:
#   LEFT  (60%) — PDF toolbar  +  Chat  +  message input
#   RIGHT (40%) — LangGraph Flow  +  Execution Trace  +  State Inspector
# ═══════════════════════════════════════════════════════════════════════════════

WELCOME = (
    "Hello! I'm your **LangGraph PDF Chatbot**.\n\n"
    "After every message the **right panel** updates live:\n"
    "- **LangGraph Flow** — active node highlighted in purple\n"
    "- **Execution Trace** — `START → router → node → END` path\n"
    "- **State Inspector** — live `ChatState` JSON\n\n"
    "Upload a PDF on the left, or just start chatting!"
)

CSS = """
body, .gradio-container { font-family: 'Inter', system-ui, sans-serif !important; }

.app-header {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%);
    border-radius: 14px; padding: 18px 22px; margin-bottom: 14px;
}
.app-header * { color: #fff !important; margin: 0 !important; }
.app-header h1 { font-size: 20px; font-weight: 700; margin-bottom: 3px !important; }
.app-header p  { font-size: 12px; opacity: .82; }

.pdf-toolbar {
    background: #fff; border: 1.5px solid #e2e8f0;
    border-radius: 12px; padding: 12px 14px; margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
}
.pdf-label {
    font-size: 10px; font-weight: 700; color: #94a3b8;
    text-transform: uppercase; letter-spacing: .08em; margin-bottom: 8px;
}
.rc {
    background: #fff; border: 1.5px solid #e2e8f0;
    border-radius: 12px; padding: 12px 14px; margin-bottom: 10px;
    box-shadow: 0 1px 3px rgba(0,0,0,.05);
    overflow: hidden; box-sizing: border-box; width: 100%;
}
.rc-title {
    font-size: 10px; font-weight: 700; color: #64748b;
    text-transform: uppercase; letter-spacing: .08em;
    display: flex; align-items: center; gap: 6px; margin-bottom: 10px;
}
.rc-dot { width: 7px; height: 7px; border-radius: 50%; display: inline-block; }
"""

with gr.Blocks(title="LangGraph PDF Chatbot") as demo:

    pdf_state = gr.State({"pdf_text": "", "has_pdf": False})

    gr.HTML("""
    <div class="app-header">
      <h1>LangGraph PDF Chatbot</h1>
      <p>Educational GUI — see exactly which node runs and how state changes, every turn.</p>
    </div>""")

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1 — CHAT  (two-column)
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("💬 Chat"):
            with gr.Row(equal_height=False):

                # ── LEFT (60%) — PDF toolbar + Chat ───────────────────────────
                with gr.Column(scale=3, min_width=400):

                    with gr.Group(elem_classes="pdf-toolbar"):
                        gr.HTML("<div class='pdf-label'>📄 PDF Document</div>")
                        with gr.Row(equal_height=True):
                            pdf_upload = gr.File(
                                label="Upload PDF",
                                file_types=[".pdf"],
                                type="filepath",
                                scale=3,
                            )
                            with gr.Column(scale=2, min_width=170):
                                pdf_status = gr.Textbox(
                                    value="No PDF loaded",
                                    label="Status",
                                    interactive=False,
                                    max_lines=1,
                                )
                                with gr.Row():
                                    remove_btn = gr.Button("Remove PDF", variant="secondary", size="sm")
                                    clear_btn  = gr.Button("New Chat",   variant="secondary", size="sm")

                        gr.HTML("<div style='height:6px'></div>")
                        web_search_toggle = gr.Checkbox(
                            label="Enable Web Search (Tavily)  — searches the web when no PDF is loaded",
                            value=False,
                            container=False,
                        )

                    chatbot = gr.Chatbot(
                        value=[_bot(WELCOME)],
                        show_label=False,
                        height=490,
                        layout="bubble",
                    )

                    with gr.Row():
                        user_input = gr.Textbox(
                            placeholder="Ask anything about the PDF, or just chat…",
                            show_label=False, scale=5, container=False, autofocus=True,
                        )
                        send_btn = gr.Button("Send ➤", variant="primary", scale=1)

                # ── RIGHT (40%) — LangGraph debug panels ──────────────────────
                with gr.Column(scale=2, min_width=300):

                    with gr.Group(elem_classes="rc"):
                        gr.HTML(
                            "<div class='rc-title'>"
                            "<span class='rc-dot' style='background:#4f46e5'></span>"
                            "LangGraph Flow"
                            "<span style='margin-left:auto;font-size:9px;color:#94a3b8;"
                            "font-weight:400;text-transform:none;letter-spacing:0'>"
                            "purple = node that ran</span></div>"
                        )
                        graph_viz = gr.HTML(value=build_graph_html())

                    with gr.Group(elem_classes="rc"):
                        gr.HTML(
                            "<div class='rc-title'>"
                            "<span class='rc-dot' style='background:#f59e0b'></span>"
                            "Execution Trace</div>"
                        )
                        trace_display = gr.HTML(value=build_trace_html([], ""))

                    with gr.Accordion("🗂 State Inspector — ChatState JSON", open=False):
                        state_display = gr.HTML(
                            value=build_state_html(
                                {"messages": [], "pdf_text": "", "has_pdf": False, "action": "chat"}
                            )
                        )

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2 — LEARN LANGGRAPH
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("📚 Learn LangGraph"):
            gr.HTML(value=LEARN_HTML)

    # ── Event wiring ──────────────────────────────────────────────────────────
    _co = [chatbot, pdf_state, user_input,  graph_viz, trace_display, state_display]
    _uo = [chatbot, pdf_state, pdf_status,  graph_viz, trace_display, state_display]
    _ro = [chatbot, pdf_state, pdf_status,  graph_viz, trace_display, state_display]

    send_btn.click(   fn=handle_chat,       inputs=[user_input, chatbot, pdf_state, web_search_toggle], outputs=_co)
    user_input.submit(fn=handle_chat,       inputs=[user_input, chatbot, pdf_state, web_search_toggle], outputs=_co)
    pdf_upload.upload(fn=handle_pdf_upload, inputs=[pdf_upload,  chatbot, pdf_state],                   outputs=_uo)
    remove_btn.click( fn=handle_remove_pdf, inputs=[chatbot],                                           outputs=_ro)
    clear_btn.click(  fn=handle_clear,      inputs=[],                                                  outputs=_ro)


# ═══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True,
        ssr_mode=False,
        css=CSS,
    )
