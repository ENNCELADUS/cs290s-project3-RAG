import { useCallback, useReducer } from "react";
import { askQuestion } from "../api/client";
import type { AnswerResponse, ChatSettings, HealthResponse, Message } from "../types";

export interface ChatState {
  messages: Message[];
  isLoading: boolean;
  settings: ChatSettings;
  systemStatus: HealthResponse | null;
  samples: string[];
  error: string | null;
}

type Action =
  | { type: "ADD_USER_MESSAGE"; payload: Message }
  | { type: "ADD_ASSISTANT_MESSAGE"; payload: Message }
  | { type: "SET_LOADING"; payload: boolean }
  | { type: "SET_ERROR"; payload: string | null }
  | { type: "UPDATE_SETTINGS"; payload: Partial<ChatSettings> }
  | { type: "SET_STATUS"; payload: HealthResponse }
  | { type: "SET_SAMPLES"; payload: string[] };

const initialState: ChatState = {
  messages: [],
  isLoading: false,
  settings: {
    mode: "hybrid",
    topK: 5,
    retrievalOnly: false,
  },
  systemStatus: null,
  samples: [],
  error: null,
};

function reducer(state: ChatState, action: Action): ChatState {
  switch (action.type) {
    case "ADD_USER_MESSAGE":
      return { ...state, messages: [...state.messages, action.payload], error: null };
    case "ADD_ASSISTANT_MESSAGE":
      return { ...state, messages: [...state.messages, action.payload], isLoading: false };
    case "SET_LOADING":
      return { ...state, isLoading: action.payload };
    case "SET_ERROR":
      return { ...state, error: action.payload, isLoading: false };
    case "UPDATE_SETTINGS":
      return { ...state, settings: { ...state.settings, ...action.payload } };
    case "SET_STATUS":
      return { ...state, systemStatus: action.payload };
    case "SET_SAMPLES":
      return { ...state, samples: action.payload };
  }
}

function makeId(): string {
  return Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function responseToMessage(response: AnswerResponse): Message {
  return {
    id: makeId(),
    role: "assistant",
    content: response.status === "retrieval_only" ? "" : response.answer,
    sources: response.sources,
    timing: response.timing ?? undefined,
    status: response.status,
    timestamp: Date.now(),
  };
}

export function useChat() {
  const [state, dispatch] = useReducer(reducer, initialState);

  const send = useCallback(
    async (query: string) => {
      const userMsg: Message = {
        id: makeId(),
        role: "user",
        content: query,
        timestamp: Date.now(),
      };
      dispatch({ type: "ADD_USER_MESSAGE", payload: userMsg });
      dispatch({ type: "SET_LOADING", payload: true });

      try {
        const response = await askQuestion({
          query,
          mode: state.settings.mode,
          top_k: state.settings.topK,
          retrieval_only: state.settings.retrievalOnly,
        });
        dispatch({ type: "ADD_ASSISTANT_MESSAGE", payload: responseToMessage(response) });
      } catch (err) {
        const msg = err instanceof Error ? err.message : "Unknown error";
        dispatch({ type: "SET_ERROR", payload: msg });
      }
    },
    [state.settings],
  );

  const updateSettings = useCallback((patch: Partial<ChatSettings>) => {
    dispatch({ type: "UPDATE_SETTINGS", payload: patch });
  }, []);

  const setStatus = useCallback((health: HealthResponse) => {
    dispatch({ type: "SET_STATUS", payload: health });
  }, []);

  const setSamples = useCallback((questions: string[]) => {
    dispatch({ type: "SET_SAMPLES", payload: questions });
  }, []);

  return { state, send, updateSettings, setStatus, setSamples };
}
