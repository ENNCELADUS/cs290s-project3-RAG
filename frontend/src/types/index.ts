export type RetrievalMode = "hybrid" | "dense" | "bm25";
export type AnswerStatus = "answered" | "insufficient_evidence" | "retrieval_only";

export interface QueryRequest {
  query: string;
  mode: RetrievalMode;
  top_k: number;
  retrieval_only: boolean;
}

export interface Source {
  source_id: number;
  title: string | null;
  url: string;
  chunk_id: number;
  document_id: number;
  snippet: string;
  score: number | null;
}

export interface Timing {
  retrieval_s: number;
  generation_s: number;
  total_s: number;
}

export interface AnswerResponse {
  query: string;
  mode: string;
  status: AnswerStatus;
  answer: string;
  sources: Source[];
  timing: Timing | null;
}

export interface HealthResponse {
  status: string;
  mode: string;
  artifacts_loaded: boolean;
  generator_loaded: boolean;
}

export interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  timing?: Timing;
  status?: AnswerStatus;
  timestamp: number;
}

export interface ChatSettings {
  mode: RetrievalMode;
  topK: number;
  retrievalOnly: boolean;
}
