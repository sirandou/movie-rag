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


def make_hitl_interceptor(
    signal_q: queue.Queue,
    event: threading.Event,
    response_q: queue.Queue,
):
    """
    Returns a custom input() that signals the Streamlit UI and blocks until
    the user responds. Because the agent now encodes choice + refinement text
    in a single input() call, this is always exactly one block-and-return.
    """
    def hitl_input(prompt: str = "") -> str:
        signal_q.put(("hitl_waiting", prompt))
        event.clear()
        event.wait()
        return response_q.get()

    return hitl_input


def run_agent(
    agent,
    question: str,
    out_q: queue.Queue,
    signal_q: queue.Queue,
    event: threading.Event,
    response_q: queue.Queue,
):
    """Run agent.query() in a background thread, streaming stdout to out_q."""
    stream = QueueStream(out_q)
    old_stdout = sys.stdout
    sys.stdout = stream

    original_input = builtins.input
    builtins.input = make_hitl_interceptor(signal_q, event, response_q)

    try:
        result = agent.query(question)
        out_q.put(("result", result))
    except Exception as exc:
        out_q.put(("error", str(exc)))
    finally:
        sys.stdout = old_stdout
        builtins.input = original_input
        out_q.put(("done", None))


def start_query(question: str):
    """
    Create fresh sync objects, reset display state, and launch the background
    agent thread. Falls through immediately — the state machine below picks it up.
    """
    out_q = queue.Queue()
    hitl_signal_q = queue.Queue()
    hitl_event = threading.Event()
    hitl_event.set()  # start unblocked so agent runs freely until first input()
    hitl_response_q = queue.Queue()

    st.session_state.out_q = out_q
    st.session_state.hitl_signal_q = hitl_signal_q
    st.session_state.hitl_event = hitl_event
    st.session_state.hitl_response_q = hitl_response_q

    st.session_state.accumulated_trace = []
    st.session_state.final_answer = None
    st.session_state.hitl_pending = False
    st.session_state.hitl_prompt = ""
    st.session_state.query_running = True

    thread = threading.Thread(
        target=run_agent,
        args=(
            st.session_state.agent,
            question,
            out_q,
            hitl_signal_q,
            hitl_event,
            hitl_response_q,
        ),
        daemon=True,
    )
    st.session_state.agent_thread = thread
    thread.start()


def respond_hitl(choice: str, refinement: str = ""):
    """
    Send the human's response to the waiting agent thread and resume execution.
    For choice "1", encodes both choice and refinement as "1|<text>" in one string.
    """
    response = f"1|{refinement}" if choice == "1" else choice
    st.session_state.hitl_response_q.put(response)
    st.session_state.hitl_event.set()
    st.session_state.hitl_pending = False
    st.session_state.hitl_prompt = ""
    st.rerun()


def show_hitl_ui():
    """Render the human-intervention panel when the agent is waiting for input."""
    st.warning("⚠️ Human intervention required — the agent needs your guidance to continue.")

    with st.expander("Execution trace so far", expanded=True):
        if st.session_state.accumulated_trace:
            st.code("".join(st.session_state.accumulated_trace), language=None)
        else:
            st.caption("No trace output yet.")

    st.subheader("What would you like to do?")

    refinement_text = st.text_input(
        "Refinement query (used only if you choose option 1)",
        key="hitl_refinement_input",
        placeholder="Describe how to refine the failed step…",
    )

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        if st.button("1 — Refine query", use_container_width=True, type="primary"):
            if not refinement_text.strip():
                st.warning("Enter a refinement query above before choosing this option.")
            else:
                respond_hitl("1", refinement_text.strip())
    with col2:
        if st.button("2 — Skip step", use_container_width=True):
            respond_hitl("2")
    with col3:
        if st.button("3 — Continue", use_container_width=True):
            respond_hitl("3")
    with col4:
        if st.button("4 — Replan", use_container_width=True):
            respond_hitl("4")
    with col5:
        if st.button("5 — Stop", use_container_width=True, type="secondary"):
            respond_hitl("5")


def run_polling_loop():
    """
    Drain out_q in a blocking loop, updating the trace display in real time.
    On a HITL signal or completion, updates session_state and calls st.rerun()
    to hand control back to Streamlit's render cycle.
    """
    out_q = st.session_state.out_q
    hitl_signal_q = st.session_state.hitl_signal_q

    with st.expander("Execution trace", expanded=True):
        trace_placeholder = st.empty()

    if st.session_state.accumulated_trace:
        trace_placeholder.code(
            "".join(st.session_state.accumulated_trace), language=None
        )

    with st.spinner("Agent is running…"):
        while True:
            # Check for HITL signal before blocking on out_q
            try:
                signal_type, signal_data = hitl_signal_q.get_nowait()
                if signal_type == "hitl_waiting":
                    st.session_state.hitl_pending = True
                    st.session_state.hitl_prompt = signal_data
                    st.rerun()
            except queue.Empty:
                pass

            try:
                msg_type, msg_data = out_q.get(timeout=0.05)
            except queue.Empty:
                continue

            if msg_type == "text":
                st.session_state.accumulated_trace.append(msg_data)
                trace_placeholder.code(
                    "".join(st.session_state.accumulated_trace), language=None
                )

            elif msg_type == "result":
                st.session_state.final_answer = msg_data

            elif msg_type == "error":
                st.error(f"Agent error: {msg_data}")
                st.session_state.query_running = False
                st.rerun()

            elif msg_type == "done":
                st.session_state.query_running = False
                st.rerun()


def display_answer():
    """Render the final answer and optional plan breakdown."""
    result = st.session_state.final_answer
    if not result:
        return
    answer = result.get("answer", "")
    if answer:
        st.subheader("Answer")
        st.markdown(answer)

    plan = result.get("plan", [])
    if plan:
        with st.expander("Execution plan", expanded=False):
            for i, step in enumerate(plan, 1):
                st.markdown(f"{i}. {step}")


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
if "query_running" not in st.session_state:
    st.session_state.query_running = False
if "hitl_pending" not in st.session_state:
    st.session_state.hitl_pending = False
if "hitl_prompt" not in st.session_state:
    st.session_state.hitl_prompt = ""
if "accumulated_trace" not in st.session_state:
    st.session_state.accumulated_trace = []
if "final_answer" not in st.session_state:
    st.session_state.final_answer = None
if "out_q" not in st.session_state:
    st.session_state.out_q = None
if "agent_thread" not in st.session_state:
    st.session_state.agent_thread = None
if "hitl_signal_q" not in st.session_state:
    st.session_state.hitl_signal_q = None
if "hitl_event" not in st.session_state:
    st.session_state.hitl_event = None
if "hitl_response_q" not in st.session_state:
    st.session_state.hitl_response_q = None

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
            "This agent pauses when it cannot complete a step automatically "
            "and asks for your guidance. You can refine the query, skip a step, "
            "continue with current results, replan, or stop."
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
        if st.session_state.query_running:
            st.warning("A query is already running. Please wait.")
        else:
            start_query(question.strip())

    elif submitted:
        st.warning("Please enter a question.")

    # --- State machine ---

    if st.session_state.hitl_pending:
        show_hitl_ui()

    elif st.session_state.query_running:
        run_polling_loop()

    if st.session_state.final_answer and not st.session_state.query_running:
        display_answer()
