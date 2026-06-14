import type { AnswerResponse, HealthResponse, QueryRequest } from "../types";

const BASE = "/api";

async function request<T>(path: string, options?: RequestInit): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    const text = await res.text().catch(() => "Unknown error");
    throw new Error(`API ${res.status}: ${text}`);
  }
  return res.json();
}

export async function askQuestion(req: QueryRequest): Promise<AnswerResponse> {
  return request<AnswerResponse>("/ask", {
    method: "POST",
    body: JSON.stringify(req),
  });
}

export async function getHealth(): Promise<HealthResponse> {
  return request<HealthResponse>("/health");
}

export async function fetchHealth(): Promise<HealthResponse> {
  return getHealth();
}

export async function getSampleQuestions(): Promise<string[]> {
  const data = await request<{ questions: string[] }>("/samples");
  return data.questions;
}

export async function fetchSamples(): Promise<{ questions: string[] }> {
  return request<{ questions: string[] }>("/samples");
}
