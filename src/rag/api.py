from __future__ import annotations

import argparse
import traceback
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .api_models import (
    AnswerResponse,
    HealthResponse,
    QueryRequest,
    SampleQuestionsResponse,
    SourceResponse,
    TimingResponse,
)
from .generate import RagAnswerer, RagAnswerResult
from .index import DEFAULT_BM25, DEFAULT_CHUNK_INDEX, DEFAULT_DB, DEFAULT_FAISS, DEFAULT_REPORT
from .retrieve import HybridRetrievalResult, Retriever

SAMPLE_QUESTIONS = [
    "上海科技大学一共有几个学院？",
    "《深度学习》这门课的任课老师是谁？",
    "计算机科学与技术专业需要修满多少学分才能毕业？",
    "信息学院专业型硕士与学术型硕士的培养方案有什么不同？",
    "我想做机器人方向，有哪些导师可以推荐？",
    "Which SIST faculty work on robotics?",
]

_state: _AppState | None = None


class _AppState:
    def __init__(
        self,
        *,
        db_path: Path,
        bm25_path: Path,
        faiss_path: Path,
        chunk_index_path: Path,
        report_path: Path,
        dense_model: str | None = None,
        model_path: Path | None = None,
        device: str = "auto",
    ) -> None:
        self.retriever: Retriever | None = None
        self.answerer: RagAnswerer | None = None
        self.init_error: str | None = None
        self.mode_label: str = "unavailable"

        print("[DEBUG] _AppState: creating Retriever...", flush=True)
        try:
            self.retriever = Retriever.from_paths(
                db_path=db_path,
                bm25_path=bm25_path,
                faiss_path=faiss_path,
                chunk_index_path=chunk_index_path,
                report_path=report_path,
                dense_model=dense_model,
            )
            self.mode_label = "retrieval_only"
            print("[DEBUG] _AppState: Retriever OK", flush=True)
        except FileNotFoundError as exc:
            self.init_error = f"RAG artifacts not found: {exc}"
            print(f"[DEBUG] _AppState: Retriever FAILED: {exc}", flush=True)
            return

        if model_path is not None and model_path.exists():
            print(f"[DEBUG] _AppState: creating RagAnswerer with model_path={model_path}", flush=True)
            try:
                self.answerer = RagAnswerer(
                    self.retriever,
                    model_path=model_path,
                    device=device,
                )
                self.mode_label = "full_rag"
                print("[DEBUG] _AppState: RagAnswerer OK", flush=True)
            except Exception as exc:
                self.init_error = f"Model load failed: {exc}"
                self.mode_label = "retrieval_only"
                print(f"[DEBUG] _AppState: RagAnswerer FAILED: {exc}", flush=True)
        else:
            print("[DEBUG] _AppState: no model_path, skipping RagAnswerer", flush=True)


def _build_app(state_instance: _AppState) -> FastAPI:
    global _state
    _state = state_instance

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    app = FastAPI(
        title="CS290S RAG API",
        description="Retrieval-augmented QA for ShanghaiTech/SIST",
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/api/health", response_model=HealthResponse)
    async def health():
        assert _state is not None
        return HealthResponse(
            status="ok" if _state.retriever else "error",
            mode=_state.mode_label,
            artifacts_loaded=_state.retriever is not None,
            generator_loaded=_state.answerer is not None,
        )

    @app.get("/api/samples", response_model=SampleQuestionsResponse)
    async def samples():
        return SampleQuestionsResponse(questions=SAMPLE_QUESTIONS)

    @app.post("/api/ask", response_model=AnswerResponse)
    async def ask(req: QueryRequest):
        assert _state is not None
        if _state.retriever is None:
            raise HTTPException(status_code=503, detail=_state.init_error or "Retriever unavailable")

        try:
            if not req.retrieval_only and _state.answerer is not None:
                return _handle_full_rag(req)
            return _handle_retrieval_only(req)
        except Exception:
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=traceback.format_exc())

    dist_path = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"
    if dist_path.is_dir():
        app.mount("/", StaticFiles(directory=str(dist_path), html=True), name="spa")

    return app


def _handle_full_rag(req: QueryRequest) -> AnswerResponse:
    assert _state is not None and _state.answerer is not None
    mode = "dense" if req.mode == "dense" else "hybrid"
    result: RagAnswerResult = _state.answerer.answer(req.query, mode=mode, top_k=req.top_k)

    sources = [
        SourceResponse(
            source_id=s.source_id,
            title=s.title,
            url=s.url,
            chunk_id=s.chunk_id,
            document_id=s.document_id,
            snippet=s.snippet,
            score=None,
        )
        for s in result.sources
    ]

    return AnswerResponse(
        query=result.query,
        mode=result.mode,
        status=result.status,
        answer=result.answer,
        sources=sources,
        timing=TimingResponse(
            retrieval_s=result.timing.retrieval_s,
            generation_s=result.timing.generation_s,
            total_s=result.timing.total_s,
        ),
    )


def _handle_retrieval_only(req: QueryRequest) -> AnswerResponse:
    assert _state is not None and _state.retriever is not None
    import time

    started = time.perf_counter()
    retrieval_result = _state.retriever.retrieve(req.query, mode=req.mode, top_k=req.top_k)
    elapsed = time.perf_counter() - started

    if isinstance(retrieval_result, HybridRetrievalResult):
        hits = retrieval_result.hits
    else:
        hits = retrieval_result

    sources = [
        SourceResponse(
            source_id=i + 1,
            title=hit.title,
            url=hit.url or "",
            chunk_id=hit.chunk_id,
            document_id=hit.document_id,
            snippet=hit.snippet,
            score=hit.score,
        )
        for i, hit in enumerate(hits)
    ]

    return AnswerResponse(
        query=req.query,
        mode=req.mode,
        status="retrieval_only",
        answer="",
        sources=sources,
        timing=TimingResponse(
            retrieval_s=elapsed,
            generation_s=0.0,
            total_s=elapsed,
        ),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Launch the RAG FastAPI server.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--bm25", type=Path, default=DEFAULT_BM25)
    parser.add_argument("--faiss", type=Path, default=DEFAULT_FAISS)
    parser.add_argument("--chunk-index", type=Path, default=DEFAULT_CHUNK_INDEX)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--dense-model", default=None)
    parser.add_argument("--model-path", type=Path, default=None)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    state_instance = _AppState(
        db_path=args.db,
        bm25_path=args.bm25,
        faiss_path=args.faiss,
        chunk_index_path=args.chunk_index,
        report_path=args.report,
        dense_model=args.dense_model,
        model_path=args.model_path,
        device=args.device,
    )
    print("[DEBUG] main: _AppState created", flush=True)

    app = _build_app(state_instance)
    print("[DEBUG] main: FastAPI app built, starting uvicorn...", flush=True)
    uvicorn.run(app, host=args.host, port=args.port)
    print("[DEBUG] main: uvicorn exited", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
