from __future__ import annotations

import argparse
import html
import traceback
from pathlib import Path
from typing import Any

import gradio as gr

from .generate import AnswerSource, RagAnswerResult, RagAnswerer
from .index import DEFAULT_BM25, DEFAULT_CHUNK_INDEX, DEFAULT_DB, DEFAULT_FAISS, DEFAULT_REPORT
from .retrieve import HybridRetrievalResult, OptimizedRetrievalHit, RetrievalHit, Retriever

_AnswerResult = RagAnswerResult | None
_RetrievalResult = list[RetrievalHit] | HybridRetrievalResult | None

SAMPLE_QUESTIONS = [
    "上海科技大学一共有几个学院？",
    "《深度学习》这门课的任课老师是谁？",
    "计算机科学与技术专业需要修满多少学分才能毕业？",
    "信息学院专业型硕士与学术型硕士的培养方案有什么不同？",
    "我想做机器人方向，有哪些导师可以推荐？",
    "Which SIST faculty work on robotics?",
]

_APP_TITLE = "CS290S Project 3: ShanghaiTech / SIST RAG"
_APP_DESCRIPTION = (
    "Retrieval-augmented QA system using official ShanghaiTech and SIST sources. "
    "Self-hosted models only — no commercial LLM API calls."
)

_CSS = """
footer {display: none !important;}
.status-row {font-size: 0.85em; color: #666;}
.source-table {width: 100%; border-collapse: collapse; font-size: 0.9em;}
.source-table th {text-align: left; padding: 4px 8px; background: #f0f0f0;}
.source-table td {padding: 4px 8px; vertical-align: top; border-bottom: 1px solid #eee;}
.source-table .rank {width: 30px; text-align: center; font-weight: bold; color: #555;}
.source-table .title {font-weight: 600;}
.source-table .url {font-size: 0.85em; color: #888; word-break: break-all;}
.source-table .score {width: 60px; text-align: right; font-variant-numeric: tabular-nums;}
.source-table .snippet {font-size: 0.9em; max-width: 400px;}
.insufficient {border-left: 4px solid #e8a838; padding: 8px 12px; background: #fffaf0;}
.error-box {border-left: 4px solid #d04444; padding: 8px 12px; background: #fff5f5;}
"""


class AppState:
    def __init__(
        self,
        *,
        db_path: Path | None = None,
        bm25_path: Path | None = None,
        faiss_path: Path | None = None,
        chunk_index_path: Path | None = None,
        report_path: Path | None = None,
        dense_model: str | None = None,
        model_path: Path | None = None,
        device: str = "auto",
    ) -> None:
        self.db_path = db_path or DEFAULT_DB
        self.bm25_path = bm25_path or DEFAULT_BM25
        self.faiss_path = faiss_path or DEFAULT_FAISS
        self.chunk_index_path = chunk_index_path or DEFAULT_CHUNK_INDEX
        self.report_path = report_path or DEFAULT_REPORT
        self.dense_model = dense_model
        self.model_path = model_path
        self.device = device

        self.retriever: Retriever | None = None
        self.answerer: RagAnswerer | None = None
        self.init_error: str | None = None
        self.mode_label: str = "retrieval-only"

        self._try_init()

    def _try_init(self) -> None:
        try:
            self.retriever = Retriever.from_paths(
                db_path=self.db_path,
                bm25_path=self.bm25_path,
                faiss_path=self.faiss_path,
                chunk_index_path=self.chunk_index_path,
                report_path=self.report_path,
                dense_model=self.dense_model,
            )
        except FileNotFoundError as exc:
            self.init_error = f"RAG artifacts not found: {exc}\n\nRun `uv run rag-build-db` and `uv run rag-build-index` to build them."
            return

        if self.model_path is not None and self.model_path.exists():
            try:
                self.answerer = RagAnswerer(
                    self.retriever,
                    model_path=self.model_path,
                    device=self.device,
                )
                self.mode_label = "full RAG"
            except Exception as exc:
                self.init_error = (
                    f"Model found at {self.model_path} but failed to load: {exc}\n\nFalling back to retrieval-only mode."
                )
                self.mode_label = "retrieval-only (model load failed)"
        else:
            if self.model_path is not None:
                self.init_error = (
                    f"Model path {self.model_path} not found. Running in retrieval-only mode.\n\n"
                    "Place a local Qwen3-4B snapshot and restart to enable answer generation."
                )

    def has_artifacts(self) -> bool:
        return self.retriever is not None

    def has_generator(self) -> bool:
        return self.answerer is not None


def _build_status_html(state: AppState) -> str:
    parts: list[str] = []
    parts.append(f'<div class="status-row">')
    parts.append(f'Mode: <strong>{html.escape(state.mode_label)}</strong>')
    parts.append(" &nbsp;|&nbsp; ")
    if state.mode_label == "full RAG":
        parts.append(f'Generator: <strong>{html.escape(str(state.model_path))}</strong>')
    else:
        parts.append("Generator: not loaded")
    parts.append("</div>")
    return "\n".join(parts)


def _build_sources_html(sources: list[AnswerSource]) -> str:
    if not sources:
        return "<p><em>No sources available.</em></p>"
    rows: list[str] = []
    rows.append('<table class="source-table">')
    rows.append("<thead><tr><th class='rank'>#</th><th class='title'>Title</th><th class='url'>URL</th><th class='score'>Score</th><th class='snippet'>Snippet</th></tr></thead>")
    rows.append("<tbody>")
    for source in sources:
        title = html.escape(source.title or "(untitled)")
        url = html.escape(source.url)
        rows.append(
            "<tr>"
            f"<td class='rank'>{source.source_id}</td>"
            f"<td class='title'>{title}</td>"
            f"<td class='url'><a href='{url}' target='_blank' rel='noopener'>{url}</a></td>"
            f"<td class='score'>&mdash;</td>"
            f"<td class='snippet'>{html.escape(source.snippet[:200])}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def _build_sources_html_from_hits(hits: list[RetrievalHit] | list[OptimizedRetrievalHit]) -> str:
    if not hits:
        return "<p><em>No sources available.</em></p>"
    rows: list[str] = []
    rows.append('<table class="source-table">')
    rows.append("<thead><tr><th class='rank'>#</th><th class='title'>Title</th><th class='url'>URL</th><th class='score'>Score</th><th class='snippet'>Snippet</th></tr></thead>")
    rows.append("<tbody>")
    for hit in hits:
        title = html.escape(hit.title or "(untitled)")
        url = html.escape(hit.url or "")
        score = f"{hit.score:.4f}"
        snippet = html.escape(hit.snippet[:200])
        rows.append(
            "<tr>"
            f"<td class='rank'>{hit.rank}</td>"
            f"<td class='title'>{title}</td>"
            f"<td class='url'><a href='{url}' target='_blank' rel='noopener'>{url}</a></td>"
            f"<td class='score'>{score}</td>"
            f"<td class='snippet'>{snippet}</td>"
            "</tr>"
        )
    rows.append("</tbody></table>")
    return "\n".join(rows)


def handle_query(
    query: str,
    mode: str,
    top_k: int,
    retrieval_only: bool,
    state: AppState,
) -> tuple[str, str, str]:
    if not query.strip():
        return "", "<p><em>Enter a question above and click Ask.</em></p>", _build_status_html(state)

    if state.retriever is None:
        error_msg = state.init_error or "Retriever is not available."
        return (
            f'<div class="error-box"><strong>Setup required</strong><br>{html.escape(error_msg)}</div>',
            "<p><em>No sources available.</em></p>",
            _build_status_html(state),
        )

    retrieval_mode = mode if mode in ("bm25", "dense", "hybrid") else "hybrid"

    try:
        if not retrieval_only and state.answerer is not None:
            result = state.answerer.answer(query, mode="dense" if retrieval_mode == "dense" else "hybrid", top_k=top_k)
            return _format_full_result(result), _build_sources_html(result.sources), _build_status_html(state)

        retrieval_result = state.retriever.retrieve(query, mode=retrieval_mode, top_k=top_k)
        return _format_retrieval_result(query, retrieval_result, retrieval_mode), _build_sources_from_retrieval(retrieval_result), _build_status_html(state)

    except Exception:
        return (
            f'<div class="error-box"><strong>Error</strong><br><pre>{html.escape(traceback.format_exc())}</pre></div>',
            "<p><em>No sources available.</em></p>",
            _build_status_html(state),
        )


def _format_full_result(result: _AnswerResult) -> str:
    if result is None:
        return '<div class="insufficient"><strong>No answer generated.</strong></div>'
    if result.status == "insufficient_evidence":
        return f'<div class="insufficient"><strong>Insufficient Evidence</strong><br>{html.escape(result.answer)}</div>'
    answer_escaped = html.escape(result.answer).replace("\n", "<br>")
    header = f"**Answer** ({result.mode}, {result.timing.total_s:.2f}s)"
    return f"{header}\n\n{answer_escaped}"


def _format_retrieval_result(query: str, result: _RetrievalResult, mode: str) -> str:
    if result is None:
        return '<div class="insufficient"><strong>No results returned.</strong></div>'
    hits = result.hits if isinstance(result, HybridRetrievalResult) else result
    if not hits:
        return '<div class="insufficient"><strong>No results found.</strong> Try a different query or check the knowledge base coverage.</div>'
    lines: list[str] = [f"**Retrieved {len(hits)} context(s)** ({mode} mode)", ""]
    for hit in hits:
        title = hit.title or "(untitled)"
        snippet = hit.snippet.replace("\n", " ")
        url = hit.url or ""
        score_str = f"{hit.score:.4f}"
        lines.append(f"**[{hit.rank}]** {html.escape(title)}")
        lines.append(f"> score: {score_str} | {html.escape(snippet[:180])}")
        if url:
            lines.append(f"> [{html.escape(url[:80])}]({html.escape(url)})")
        lines.append("")
    return "\n".join(lines)


def _build_sources_from_retrieval(result: _RetrievalResult) -> str:
    if result is None:
        return "<p><em>No sources available.</em></p>"
    hits = result.hits if isinstance(result, HybridRetrievalResult) else result
    return _build_sources_html_from_hits(list(hits))


def _on_sample_click(question: str) -> tuple[str, str]:
    return question, ""


def create_app(state: AppState) -> gr.Blocks:
    with gr.Blocks(
        title=_APP_TITLE,
        css=_CSS,
        theme=gr.themes.Soft(),
    ) as demo:
        gr.Markdown(f"# {_APP_TITLE}")
        gr.Markdown(_APP_DESCRIPTION)

        status_html = gr.HTML(value=_build_status_html(state))

        init_msg = ""
        if state.init_error:
            init_msg = f'<div class="error-box"><strong>Notice</strong><br>{html.escape(state.init_error)}</div>'

        with gr.Row():
            with gr.Column(scale=4):
                question_input = gr.Textbox(
                    label="Question",
                    placeholder="Ask a question about ShanghaiTech University or SIST...",
                    lines=2,
                )
            with gr.Column(scale=1, min_width=100):
                ask_btn = gr.Button("Ask", variant="primary", size="lg")

        with gr.Row():
            mode_radio = gr.Radio(
                choices=["hybrid", "dense", "bm25"],
                value="hybrid",
                label="Retrieval Mode",
                interactive=True,
            )
            top_k_slider = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Top-K",
            )
            retrieval_only_checkbox = gr.Checkbox(
                label="Retrieval-only mode",
                value=state.answerer is None,
                interactive=state.answerer is not None,
                info="Show retrieved sources without generating an answer.",
            )

        initial_answer = init_msg or "<p><em>Enter a question above and click Ask.</em></p>"
        answer_md = gr.Markdown(value=initial_answer, elem_id="answer-panel")

        with gr.Accordion("Sources", open=False):
            sources_html = gr.HTML(value="<p><em>No sources yet.</em></p>")

        with gr.Accordion("Sample Questions", open=False):
            sample_buttons: list[gr.Button] = []
            for idx, sample_q in enumerate(SAMPLE_QUESTIONS):
                btn = gr.Button(sample_q, size="sm", variant="secondary")
                sample_buttons.append(btn)

        inputs = [question_input, mode_radio, top_k_slider, retrieval_only_checkbox]
        ask_btn.click(
            fn=lambda q, m, k, ro: handle_query(q, m, k, ro, state),
            inputs=inputs,
            outputs=[answer_md, sources_html, status_html],
        )
        question_input.submit(
            fn=lambda q, m, k, ro: handle_query(q, m, k, ro, state),
            inputs=inputs,
            outputs=[answer_md, sources_html, status_html],
        )

        for btn, sample_q in zip(sample_buttons, SAMPLE_QUESTIONS):
            btn.click(
                fn=lambda q=sample_q: (q, ""),
                inputs=[],
                outputs=[question_input, answer_md],
            ).then(
                fn=lambda q, m, k, ro: handle_query(q, m, k, ro, state),
                inputs=inputs,
                outputs=[answer_md, sources_html, status_html],
            )

    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the RAG Gradio web UI.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB, help="SQLite corpus database path.")
    parser.add_argument("--bm25", type=Path, default=DEFAULT_BM25, help="BM25 pickle payload path.")
    parser.add_argument("--faiss", type=Path, default=DEFAULT_FAISS, help="FAISS dense index path.")
    parser.add_argument("--chunk-index", type=Path, default=DEFAULT_CHUNK_INDEX, help="Chunk mapping JSONL path.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT, help="Build report JSON path.")
    parser.add_argument("--dense-model", default=None, help="Path or HF ID for the dense embedding model.")
    parser.add_argument("--model-path", type=Path, default=None, help="Local Qwen checkpoint path for answer generation.")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto", help="Device for the generator model.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host.")
    parser.add_argument("--port", type=int, default=7860, help="Server port.")
    parser.add_argument("--share", action="store_true", help="Create a Gradio public share link.")
    args = parser.parse_args(argv)

    state = AppState(
        db_path=args.db,
        bm25_path=args.bm25,
        faiss_path=args.faiss,
        chunk_index_path=args.chunk_index,
        report_path=args.report,
        dense_model=args.dense_model,
        model_path=args.model_path,
        device=args.device,
    )

    demo = create_app(state)
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
