import builtins
import io
import queue
import re
import sys
import threading
from pathlib import Path

import streamlit as st

from src.langchain.prompts import ZERO_SHOT_QA_PROMPT

sys.path.insert(0, str(Path(__file__).parent))

from src.agents.adaptive_plan_execute import AdaptivePlanExecuteAgent
from src.agents.hitl_adaptive_plan_execute import AdaptHITLPlanExecAgent
from src.agents.plan_execute import PlanExecuteAgent
from src.agents.react import ReactAgent
from src.langchain.chains.movie_rag import MovieRAGChain
from src.langchain.loaders import MoviePosterDocumentLoader
from src.retrievers.visual_retriever import VisualRetriever

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AGENT_CLASSES = {
    "ReAct": ReactAgent,
    "Plan-Execute": PlanExecuteAgent,
    "Adaptive Plan-Execute": AdaptivePlanExecuteAgent,
    "HITL Adaptive": AdaptHITLPlanExecAgent,
}

DEFAULT_PLOTS = "/Users/saghar/Desktop/movie-rag/datasets/rotten-tomatoes-reviews/prep/movie_plots.csv"
DEFAULT_REVIEWS = "/Users/saghar/Desktop/movie-rag/datasets/rotten-tomatoes-reviews/prep/reviews_w_movies_full.csv"
DEFAULT_SQL = "/Users/saghar/Desktop/movie-rag/datasets/rotten-tomatoes-reviews/prep/movies_meta.db"
DEFAULT_POSTERS = "/Users/saghar/Desktop/movie-rag/datasets/rotten-tomatoes-reviews/prep/movie_posters.csv"

ANSI_RE = re.compile(r"\033\[[0-9;]*m")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


class QueueStream(io.TextIOBase):
    """Redirects stdout writes into a queue for live Streamlit display."""

    def __init__(self, q: queue.Queue):
        self._queue = q

    def write(self, text: str) -> int:
        if text:
            self._queue.put(("text", strip_ansi(text)))
        return len(text)

    def flush(self):
        pass


def build_pipeline(cfg: dict):
    """
    Build RAG chain + visual retriever + agent.
    Runs in the main thread inside st.status so progress is visible.
    """
    with st.status("Initializing pipeline…", expanded=True) as status:
        st.write("Building RAG chain…")
        chain = MovieRAGChain(
            plots_path=cfg["plots_path"],
            reviews_path=cfg["reviews_path"],
            max_movies=cfg["max_movies"],
            use_custom_retriever=True,
            use_custom_chunk=True,
            custom_prompt=ZERO_SHOT_QA_PROMPT,
            k=5,
            initial_k=20,
            use_hyde=True,
            hyde_model="gpt-4o-mini",
            use_reranking=True,
            reranker_cfg={'type': 'llm'},
            embed_model="text-embedding-3-small",
            llm_model="gpt-4o-mini",
        )

        chain.build()
        st.write("✓ RAG chain ready")

        st.write(f"Loading {cfg['max_posters']} posters and encoding with CLIP…")
        poster_loader = MoviePosterDocumentLoader(
            cfg["poster_path"], max_movies=cfg["max_posters"]
        )
        visual_docs = poster_loader.load()
        visual_retriever = VisualRetriever(
            model_name="ViT-B/32", use_text_fusion=True, alpha=0.8
        )
        visual_retriever.add_documents(visual_docs)
        st.write(f"✓ Visual retriever ready ({len(visual_docs)} posters)")

        st.write(f"Creating {cfg['agent_type']} agent…")
        AgentClass = AGENT_CLASSES[cfg["agent_type"]]
        agent = AgentClass(
            chain,
            visual_retriever,
            cfg["sql_path"],
            cfg["reviews_path"],
            web_search_enabled=cfg["web_search"],
        )
        st.write(f"✓ {cfg['agent_type']} agent ready")

        status.update(label="Pipeline ready!", state="complete", expanded=False)

    return agent


def run_agent(agent, question: str, out_q: queue.Queue):
    """Run agent.query() in a background thread, streaming stdout to out_q."""
    stream = QueueStream(out_q)
    old_stdout = sys.stdout
    sys.stdout = stream

    # HITL agent calls input() for human choices; auto-accept (option 3) for now.
    original_input = builtins.input
    builtins.input = lambda prompt="": "3"

    try:
        result = agent.query(question)
        out_q.put(("result", result))
    except Exception as exc:
        out_q.put(("error", str(exc)))
    finally:
        sys.stdout = old_stdout
        builtins.input = original_input
        out_q.put(("done", None))


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Movie RAG Agent", page_icon="🎬", layout="wide")
st.title("🎬 Movie RAG Agent")

# ---------------------------------------------------------------------------
# Session state defaults
# ---------------------------------------------------------------------------

if "initialized" not in st.session_state:
    st.session_state.initialized = False
if "agent" not in st.session_state:
    st.session_state.agent = None

# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Configuration")

    plots_path = st.text_input("Plots CSV", DEFAULT_PLOTS)
    reviews_path = st.text_input("Reviews CSV", DEFAULT_REVIEWS)
    sql_path = st.text_input("SQLite DB", DEFAULT_SQL)
    poster_path = st.text_input("Posters CSV", DEFAULT_POSTERS)

    st.divider()

    max_movies = st.slider("Max movies", 100, 8000, 500, step=100)
    max_posters = st.slider("Max posters", 100, 6000, 1000, step=100)

    st.divider()

    agent_type = st.selectbox("Agent type", list(AGENT_CLASSES.keys()))
    web_search = st.toggle("Enable web search", value=False)

    if agent_type == "HITL Adaptive":
        st.caption(
            "⚠️ Human-in-the-loop interactions will auto-accept in this UI. "
            "Full HITL support coming soon."
        )

    st.divider()

    init_clicked = st.button(
        "🚀 Initialize Pipeline", type="primary", use_container_width=True
    )

    if st.session_state.initialized:
        st.success(f"Ready · {st.session_state.agent_type}")

# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------

if init_clicked:
    cfg = {
        "plots_path": plots_path,
        "reviews_path": reviews_path,
        "sql_path": sql_path,
        "poster_path": poster_path,
        "max_movies": max_movies,
        "max_posters": max_posters,
        "agent_type": agent_type,
        "web_search": web_search,
    }
    try:
        agent = build_pipeline(cfg)
        st.session_state.agent = agent
        st.session_state.agent_type = agent_type
        st.session_state.initialized = True
        st.rerun()
    except Exception as exc:
        st.error(f"Initialization failed: {exc}")

# ---------------------------------------------------------------------------
# Query area
# ---------------------------------------------------------------------------

if not st.session_state.initialized:
    st.info("Configure options in the sidebar, then click **🚀 Initialize Pipeline**.")
else:
    st.subheader("Ask a question")

    with st.form("query_form", clear_on_submit=False):
        question = st.text_area(
            "Question",
            placeholder="e.g. Find dark sci-fi movies with great visuals",
            height=80,
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask →", type="primary")

    if submitted and question.strip():
        out_q: queue.Queue = queue.Queue()

        thread = threading.Thread(
            target=run_agent,
            args=(st.session_state.agent, question.strip(), out_q),
            daemon=True,
        )
        thread.start()

        # --- Live trace inside an expander ---
        with st.expander("Execution trace", expanded=True):
            trace_placeholder = st.empty()

        answer_header = st.empty()
        answer_placeholder = st.empty()

        trace_lines: list[str] = []
        result = None

        while True:
            try:
                msg_type, msg_data = out_q.get(timeout=0.05)
            except queue.Empty:
                continue

            if msg_type == "text":
                trace_lines.append(msg_data)
                trace_placeholder.code("".join(trace_lines), language=None)

            elif msg_type == "result":
                result = msg_data

            elif msg_type == "error":
                st.error(f"Agent error: {msg_data}")
                break

            elif msg_type == "done":
                break

        thread.join()

        if result:
            answer = result.get("answer", "")
            answer_header.subheader("Answer")
            answer_placeholder.markdown(answer)

    elif submitted:
        st.warning("Please enter a question.")
