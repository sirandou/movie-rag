"""Streamlit UI for Movie RAG Agents.

Run with:
    streamlit run app.py
"""

import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import streamlit as st

# ---------------------------------------------------------------------------
# Constants & defaults
# ---------------------------------------------------------------------------

_PREP_DIR = Path(__file__).parent / "datasets" / "rotten-tomatoes-reviews" / "prep"
_DATA_ROOT = _PREP_DIR.parent  # .../datasets/rotten-tomatoes-reviews

DEFAULT_PLOTS_PATH = str(_PREP_DIR / "movie_plots.csv")
DEFAULT_REVIEWS_PATH = str(_PREP_DIR / "reviews_w_movies_full.csv")
DEFAULT_DB_PATH = str(_PREP_DIR / "movies_meta.db")
DEFAULT_POSTERS_CSV_PATH = str(_PREP_DIR / "movie_posters.csv")

AGENT_DESCRIPTIONS: Dict[str, str] = {
    "ReAct": (
        "Simple action-observation loop. The agent calls tools one at a time "
        "until it has enough information to answer. Fast and direct."
    ),
    "Plan-Execute": (
        "Creates a numbered multi-step plan for the query, then executes each "
        "step sequentially using the available tools."
    ),
    "Adaptive Plan-Execute": (
        "Like Plan-Execute but automatically retries failed steps, falls back "
        "to web search, and can replan when too many steps fail."
    ),
    "HITL Adaptive": (
        "Like Adaptive Plan-Execute but asks for your input when it gets "
        "completely stuck — you can refine queries, skip steps, or replan."
    ),
}

LLM_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4-turbo"]

_SETUP_INSTRUCTIONS = """
### Setup instructions

The agents require processed movie data that must be prepared before use.

**Step 1 — Download raw data** from
[Kaggle](https://www.kaggle.com/datasets/stefanoleone992/rotten-tomatoes-movies-and-critic-reviews-dataset/data)
and place the CSV files in `datasets/rotten-tomatoes-reviews/raw/`.

**Step 2 — Run data-prep notebooks** in order:
1. `notebooks/data_prep/prep_rotten_tomatoes_data.ipynb`
2. `notebooks/data_prep/create_omdb_plots_data.ipynb`
3. `notebooks/data_prep/create_omdb_posters_data.ipynb`

**Step 3 — Build the SQLite database:**
```
python src/data/sqlite_database.py
```

**Step 4 — Set your OpenAI API key:**
```
export OPENAI_API_KEY=sk-...
```

**Required files** (all inside `datasets/rotten-tomatoes-reviews/prep/`):
- `movie_plots.csv`
- `reviews_w_movies_full.csv`
- `movie_posters.csv`
- `movies_meta.db`

After completing these steps, refresh this page.
"""


# ---------------------------------------------------------------------------
# Session-state initialisation
# ---------------------------------------------------------------------------


def _init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "messages": [],          # [{role, content, plan?, step_results?}]
        "agent_running": False,
        "hitl_pending": None,    # {prompt, context} set by agent thread
        "hitl_response": None,   # {choice, refined_query} set by main thread
        "hitl_event": None,      # threading.Event for synchronisation
        "agent_thread": None,    # background Thread
        "result_holder": None,   # {"result": ..., "error": ...}
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ---------------------------------------------------------------------------
# Data validation
# ---------------------------------------------------------------------------


def _safe_data_path(path_str: str, expected_suffix: Optional[str] = None) -> Optional[Path]:
    """Validate a user-supplied path is within _DATA_ROOT.

    Returns the resolved Path if allowed, otherwise None.
    """
    if not path_str:
        return None
    try:
        candidate = Path(path_str).expanduser().resolve()
    except OSError:
        return None
    try:
        candidate.relative_to(_DATA_ROOT)
    except ValueError:
        return None
    if expected_suffix is not None and candidate.suffix.lower() != expected_suffix.lower():
        return None
    return candidate


def _missing_files(plots: str, reviews: str, db: str) -> list:
    missing = []
    for label, path_str, expected_suffix in [
        ("movie_plots.csv", plots, ".csv"),
        ("reviews_w_movies_full.csv", reviews, ".csv"),
        ("movies_meta.db", db, ".db"),
    ]:
        safe_path = _safe_data_path(path_str, expected_suffix=expected_suffix)
        if safe_path is None or not safe_path.exists():
            reason = "path not allowed" if safe_path is None else "file missing"
            missing.append(f"`{path_str}` — {label} ({reason})")
    return missing


# ---------------------------------------------------------------------------
# RAG-component loading (cached across reruns)
# ---------------------------------------------------------------------------


@st.cache_resource(show_spinner=False)
def _load_rag_components(
    plots_path: str,
    reviews_path: str,
    posters_csv_path: str,
    max_text_movies: int,
    max_poster_movies: int,
    use_hyde: bool,
    use_reranking: bool,
) -> tuple:
    """Build and cache the MovieRAGChain and VisualRetriever.

    All parameters are part of the cache key — changing any of them triggers
    a full rebuild.
    """
    from src.langchain.chains.movie_rag import MovieRAGChain
    from src.langchain.loaders import MoviePosterDocumentLoader
    from src.langchain.prompts import ZERO_SHOT_QA_PROMPT
    from src.retrievers.visual_retriever import VisualRetriever

    with st.status("Loading RAG components (one-time setup)…", expanded=True) as status:
        st.write("Initialising CLIP visual retriever…")
        visual_retriever = VisualRetriever(
            model_name="ViT-B/32",
            use_text_fusion=True,
            alpha=0.8,
        )
        if Path(posters_csv_path).exists():
            st.write(f"Encoding movie posters (up to {max_poster_movies})…")
            loader = MoviePosterDocumentLoader(
                posters_path=posters_csv_path,
                max_movies=max_poster_movies,
            )
            visual_retriever.add_documents(loader.load())

        features = []
        if use_hyde:
            features.append("HyDE")
        if use_reranking:
            features.append("reranking")
        feature_str = f" with {', '.join(features)}" if features else ""
        st.write(f"Building text RAG chain{feature_str} (up to {max_text_movies} movies)…")
        chain = MovieRAGChain(
            plots_path=plots_path,
            reviews_path=reviews_path,
            max_movies=max_text_movies,
            use_custom_retriever=True,
            use_custom_chunk=True,
            custom_prompt=ZERO_SHOT_QA_PROMPT,
            k=5,
            use_hyde=use_hyde,
            hyde_model="gpt-4o-mini",
            use_reranking=use_reranking,
            reranker_cfg={"type": "llm"},
            initial_k=20,
        )
        chain.build()

        status.update(label="RAG components ready!", state="complete", expanded=False)

    return chain, visual_retriever


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------


def _build_agent(
    agent_type: str,
    text_chain,
    visual_retriever,
    db_path: str,
    reviews_path: str,
    llm_model: str,
    temperature: float,
    web_search: bool,
    human_input_fn: Optional[Callable] = None,
):
    kwargs = dict(
        text_chain=text_chain,
        visual_retriever=visual_retriever,
        sql_database_path=db_path,
        reviews_csv_path=reviews_path,
        llm_model=llm_model,
        llm_temperature=temperature,
        web_search_enabled=web_search,
    )
    if agent_type == "ReAct":
        from src.agents.react import ReactAgent

        return ReactAgent(**kwargs)
    elif agent_type == "Plan-Execute":
        from src.agents.plan_execute import PlanExecuteAgent

        return PlanExecuteAgent(**kwargs)
    elif agent_type == "Adaptive Plan-Execute":
        from src.agents.adaptive_plan_execute import AdaptivePlanExecuteAgent

        return AdaptivePlanExecuteAgent(**kwargs)
    else:  # HITL Adaptive
        from src.agents.hitl_adaptive_plan_execute import AdaptHITLPlanExecAgent

        return AdaptHITLPlanExecAgent(**kwargs, human_input_fn=human_input_fn)


# ---------------------------------------------------------------------------
# HITL callback (executes inside the background agent thread)
# ---------------------------------------------------------------------------


def _make_hitl_callback(event: threading.Event) -> Callable:
    """Return a callback the HITL agent calls in place of ``input()``."""

    def _callback(prompt: str, context: Dict[str, Any]) -> Dict[str, Any]:
        # Tell the main Streamlit thread that input is needed.
        st.session_state.hitl_pending = {"prompt": prompt, "context": context}
        # Block until the user responds via the UI.
        event.clear()
        event.wait()
        # Consume the response and unblock.
        response = st.session_state.hitl_response or {"choice": "3", "refined_query": ""}
        st.session_state.hitl_response = None
        st.session_state.hitl_pending = None
        return response

    return _callback


# ---------------------------------------------------------------------------
# Agent runner (background thread target)
# ---------------------------------------------------------------------------


def _run_agent(agent, question: str, holder: dict) -> None:
    try:
        holder["result"] = agent.query(question, verbose=False)
        holder["error"] = None
    except Exception as exc:
        holder["result"] = None
        holder["error"] = str(exc)


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def _render_chat_history() -> None:
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            plan = msg.get("plan") or []
            step_results = msg.get("step_results") or []
            if plan:
                with st.expander(f"Execution plan ({len(plan)} steps)"):
                    for i, step in enumerate(plan, 1):
                        st.markdown(f"**{i}.** {step}")
            if step_results:
                with st.expander("Step-by-step results"):
                    for i, result in enumerate(step_results, 1):
                        preview = result[:400] + ("…" if len(result) > 400 else "")
                        st.markdown(f"**Step {i}:** {preview}")


def _render_hitl_panel(pending: dict) -> None:
    context = pending.get("context", {})
    plan = context.get("plan", [])
    step_num = context.get("step", "?")
    step_text = context.get("step_text", "")
    result_preview = context.get("result", "")

    st.info(
        f"**The agent needs your help — Step {step_num} of {len(plan)}**\n\n"
        f"The step below could not be completed automatically:"
    )

    if step_text:
        st.markdown(f"> {step_text}")

    if result_preview:
        with st.expander("Current result (from fallback search)", expanded=False):
            st.write(result_preview[:800] + ("…" if len(result_preview) > 800 else ""))

    choice = st.radio(
        "What would you like to do?",
        options=["1", "2", "3", "4", "5"],
        format_func=lambda x: {
            "1": "Refine the search query",
            "2": "Skip this step",
            "3": "Accept current result and continue",
            "4": "Replan the approach",
            "5": "Stop execution",
        }[x],
        key="hitl_choice_radio",
    )

    refined_query = ""
    if choice == "1":
        refined_query = st.text_input(
            "How should we refine the query?",
            key="hitl_refined_query_input",
        )

    if st.button("Confirm", type="primary", key="hitl_confirm_btn"):
        st.session_state.hitl_response = {
            "choice": choice,
            "refined_query": refined_query,
        }
        st.session_state.hitl_event.set()
        st.rerun()


def _finalise_result(holder: dict) -> None:
    """Move agent thread result into chat history and reset running state."""
    st.session_state.agent_running = False
    st.session_state.agent_thread = None

    if holder.get("error"):
        st.session_state.messages.append(
            {"role": "assistant", "content": f"**Error:** {holder['error']}"}
        )
    elif holder.get("result"):
        result = holder["result"]
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": result.get("answer", "*(no answer)*"),
                "plan": result.get("plan", []),
                "step_results": result.get("step_results", []),
            }
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    st.set_page_config(
        page_title="Movie Agent RAG",
        page_icon="🎬",
        layout="wide",
    )
    _init_session_state()

    # ── Sidebar ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("🎬 Movie Agent RAG")
        st.caption("Multi-agent system powered by LangGraph & OpenAI")
        st.divider()

        agent_type = st.selectbox(
            "Agent type",
            options=list(AGENT_DESCRIPTIONS.keys()),
        )
        st.caption(AGENT_DESCRIPTIONS[agent_type])
        st.divider()

        st.subheader("LLM settings")
        llm_model = st.selectbox("Model", LLM_MODELS)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.0, step=0.05)
        web_search = st.toggle(
            "Enable web search",
            value=False,
            help="Requires TAVILY_API_KEY set in your environment.",
        )
        st.divider()

        with st.expander("Indexing settings"):
            st.caption(
                "Changing these rebuilds the index from scratch. "
                "Fewer movies = faster startup, lower coverage."
            )
            max_text_movies = st.slider(
                "Max movies (text index)", 100, 8075, 500, step=100,
                help="Number of movies loaded into the plots/reviews FAISS index.",
            )
            max_poster_movies = st.slider(
                "Max movies (poster index)", 100, 6079, 1000, step=100,
                help="Number of movie posters encoded by CLIP.",
            )
            use_hyde = st.toggle("HyDE query expansion", value=True,
                                 help="Generates a hypothetical answer to improve retrieval.")
            use_reranking = st.toggle("LLM reranking", value=True,
                                      help="Re-ranks the top-20 candidates with an LLM before returning top-5.")

        with st.expander("Data paths"):
            plots_path = st.text_input("Plots CSV", DEFAULT_PLOTS_PATH)
            reviews_path = st.text_input("Reviews CSV", DEFAULT_REVIEWS_PATH)
            db_path = st.text_input("SQLite DB", DEFAULT_DB_PATH)
            posters_csv = st.text_input("Posters CSV", DEFAULT_POSTERS_CSV_PATH)

        st.divider()
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # ── Main area ──────────────────────────────────────────────────────────
    st.header("Movie Agent RAG")

    # Guard: missing data files
    missing = _missing_files(plots_path, reviews_path, db_path)
    if missing:
        st.error(
            "**Required data files are missing:**\n"
            + "\n".join(f"- {f}" for f in missing)
        )
        st.markdown(_SETUP_INSTRUCTIONS)
        return

    # Load cached RAG components
    try:
        text_chain, visual_retriever = _load_rag_components(
            plots_path, reviews_path, posters_csv,
            max_text_movies, max_poster_movies, use_hyde, use_reranking,
        )
    except Exception as exc:
        st.error(f"Failed to load RAG components: {exc}")
        return

    # Render conversation so far
    _render_chat_history()

    # ── Handle a running agent ─────────────────────────────────────────────
    if st.session_state.agent_running:
        thread: Optional[threading.Thread] = st.session_state.agent_thread

        if st.session_state.hitl_pending:
            # HITL: show the interaction panel and wait for user action
            _render_hitl_panel(st.session_state.hitl_pending)
            # Do NOT rerun automatically — let the Confirm button trigger it.
            return

        if thread and not thread.is_alive():
            # Agent finished → move result into chat history
            _finalise_result(st.session_state.result_holder)
            st.rerun()
            return

        # Still running → poll every 400 ms
        with st.spinner(f"**{agent_type}** agent is thinking…"):
            time.sleep(0.4)
        st.rerun()
        return

    # ── Chat input ─────────────────────────────────────────────────────────
    question = st.chat_input("Ask anything about movies…")
    if not question:
        return

    # Append user message immediately
    st.session_state.messages.append({"role": "user", "content": question})

    # Prepare shared state for the background thread
    result_holder: Dict[str, Any] = {"result": None, "error": None}
    event = threading.Event()
    st.session_state.result_holder = result_holder
    st.session_state.hitl_event = event
    st.session_state.agent_running = True
    st.session_state.hitl_pending = None

    human_input_fn = _make_hitl_callback(event) if agent_type == "HITL Adaptive" else None

    # Build the agent (fast — only the LangGraph compile step)
    try:
        agent = _build_agent(
            agent_type=agent_type,
            text_chain=text_chain,
            visual_retriever=visual_retriever,
            db_path=db_path,
            reviews_path=reviews_path,
            llm_model=llm_model,
            temperature=temperature,
            web_search=web_search,
            human_input_fn=human_input_fn,
        )
    except Exception as exc:
        st.session_state.agent_running = False
        st.session_state.messages.append(
            {"role": "assistant", "content": f"**Failed to initialise agent:** {exc}"}
        )
        st.rerun()
        return

    # Launch background thread
    thread = threading.Thread(
        target=_run_agent,
        args=(agent, question, result_holder),
        daemon=True,
    )
    st.session_state.agent_thread = thread
    thread.start()
    st.rerun()


if __name__ == "__main__":
    main()
