# LangGraph PDF Chatbot

An **educational** web chatbot built with **LangGraph** + **Azure OpenAI** + **Gradio**.

Upload a PDF, ask questions, and get AI-powered answers — while watching exactly how
LangGraph routes, executes nodes, and updates state at every step.

---

## Features

| Feature                   | Description                                                                    |
| ------------------------- | ------------------------------------------------------------------------------ |
| PDF Analysis              | Upload any text-based PDF → AI summarises it and generates 6 questions         |
| Document Q&A              | Every message answered using the PDF as context (simplified RAG)               |
| General Chat              | Works as a regular AI assistant when no PDF is loaded                          |
| Multi-turn memory         | Full conversation history preserved across turns                               |
| LangGraph Flow Visualizer | CSS flowchart that highlights the active node after each message               |
| Execution Trace           | Shows `START → router → node → END` path with routing reason                   |
| State Inspector           | Live JSON view of `ChatState` after every step                                 |
| Learn LangGraph tab       | 5 concept cards with code from this project (State, Node, Router, Edge, Graph) |

---

## Project Structure

```
python-langgraph-chatbot/
├── graph.py          # LangGraph workflow — State, Nodes, Router, Edges
├── app.py            # Gradio web UI — Chat tab + Learn LangGraph tab
├── .env              # Your API credentials (never commit this)
├── .env.example      # Template — copy to .env and fill in values
├── .gitignore        # Keeps .env out of git
├── requirements.txt  # Python dependencies
└── README.md
```

---

## LangGraph Architecture

```
START
  │
  └─► route_action()          ← reads state.action + state.has_pdf
        │
        ├─[action="generate_questions"] ──► generate_questions_node
        │                                         │
        ├─[action="chat" AND has_pdf=True] ──────► answer_with_context_node
        │                                         │
        └─[action="chat" AND has_pdf=False] ─────► chat_node
                                                  │
                                                 END
```

### LangGraph Key Concepts

| Concept                       | File / Location                      | What it does                                              |
| ----------------------------- | ------------------------------------ | --------------------------------------------------------- |
| **State** (`ChatState`)       | `graph.py`                           | TypedDict shared across all nodes — the "working memory"  |
| **Node** (`*_node` functions) | `graph.py`                           | Receives state, calls LLM, returns a dict of updates      |
| **Router** (`route_action`)   | `graph.py`                           | Returns a string that picks the next node                 |
| **Conditional Edge**          | `graph.py` `add_conditional_edges()` | Runs the router and follows its decision                  |
| **`add_messages`**            | `messages` field in `ChatState`      | Reducer that auto-appends messages instead of overwriting |
| **`graph.compile()`**         | bottom of `graph.py`                 | Turns the assembled graph into a runnable object          |

---

## Setup & Run

### 1. Create a virtual environment

```bash
cd python-langgraph-chatbot

# Create venv (only needed once)
python3 -m venv .venv

# Activate venv  ← run this every time you open a new terminal
source .venv/bin/activate
```

> **Why not plain `pip`?** macOS does not ship `pip` as a standalone command.
> After activating the venv you'll see `(.venv)` in your prompt — then `pip` works.

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure API credentials

```bash
cp .env.example .env   # create your .env from the template
```

Edit `.env` and fill in your Azure OpenAI values:

```env
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_DEPLOYMENT=gpt-5.2-chat
AZURE_OPENAI_API_VERSION=2025-04-01-preview
```

### 4. Run the app

```bash
source .venv/bin/activate   # activate venv if not already active
python3 app.py
```

Open **<http://localhost:7860>** in your browser.

---

## Usage

### Chat tab

1. **Upload PDF** → AI extracts text, summarises the document, suggests 6 questions
2. **Ask anything** → AI answers using the PDF as context
3. **Remove PDF** → switches back to general chat mode
4. **New Chat** → clears everything and resets state

After every message, three panels update automatically:

- **LangGraph Flow** (left sidebar) — the node that ran is highlighted in purple
- **Execution Trace** (below chat) — shows the full routing path and the reason
- **State Inspector** (below chat, collapsed by default) — shows `ChatState` as JSON

### Learn LangGraph tab

Five concept cards with colour-coded code snippets taken directly from `graph.py`:

1. **State** — the shared TypedDict
2. **Node** — plain Python functions that update state
3. **Router** — the function that picks the next node
4. **Edge** — normal vs. conditional connections
5. **Graph** — how to assemble, compile, and invoke

---

## Environment Variables

| Variable                   | Required                           | Description                            |
| -------------------------- | ---------------------------------- | -------------------------------------- |
| `AZURE_OPENAI_ENDPOINT`    | Yes                                | Base URL of your Azure OpenAI resource |
| `AZURE_OPENAI_API_KEY`     | Yes                                | Your Azure OpenAI API key              |
| `AZURE_OPENAI_DEPLOYMENT`  | No (default: `gpt-5.2-chat`)       | Deployment name in Azure portal        |
| `AZURE_OPENAI_API_VERSION` | No (default: `2025-04-01-preview`) | Azure API version                      |

---

## Tech Stack

| Library            | Version | Role                          |
| ------------------ | ------- | ----------------------------- |
| `langgraph`        | ≥ 0.2   | Graph workflow engine         |
| `langchain-openai` | ≥ 0.2   | Azure OpenAI LLM integration  |
| `langchain-core`   | ≥ 0.3   | Message types, reducers       |
| `gradio`           | ≥ 5.0   | Web UI                        |
| `pypdf`            | ≥ 4.0   | PDF text extraction           |
| `python-dotenv`    | ≥ 1.0   | Load `.env` into `os.environ` |

---

## Notes

- Works best with **text-based PDFs** (not scanned / image-based PDFs)
- PDF content is capped at **10,000 characters** to stay within the model context window;
  for very long documents consider adding chunking + vector search (full RAG pipeline)
- `.env` is listed in `.gitignore` — your API key will never be accidentally committed
