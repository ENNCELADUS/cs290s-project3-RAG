import { useState } from "react";
import { Clock } from "lucide-react";
import type { Message } from "../types";
import { useLabels } from "../i18n/labels";
import { SourceList } from "./SourceList";

interface Props {
  message: Message;
}

function renderContent(text: string, onCitationClick: (id: number) => void) {
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    const match = part.match(/^\[(\d+)\]$/);
    if (match) {
      const id = parseInt(match[1], 10);
      return (
        <span
          key={i}
          className="citation-badge mx-0.5"
          onClick={() => onCitationClick(id)}
          title={`Source ${id}`}
        >
          {id}
        </span>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

export function ChatMessage({ message }: Props) {
  const [highlightedSource, setHighlightedSource] = useState<number | null>(null);
  const t = useLabels(message.content);

  if (message.role === "user") {
    return (
      <div className="flex justify-end fade-in">
        <div className="chat-bubble-user">
          <p className="text-sm leading-relaxed whitespace-pre-wrap">{message.content}</p>
        </div>
      </div>
    );
  }

  const isInsufficient = message.status === "insufficient_evidence";
  const isRetrievalOnly = message.status === "retrieval_only";

  return (
    <div className="flex justify-start fade-in">
      <div className="chat-bubble-assistant">
        {isInsufficient && (
          <div className="flex items-start gap-2 text-yellow-700 dark:text-yellow-400 bg-yellow-50 dark:bg-yellow-900/20 rounded-lg p-2.5 mb-2">
            <span className="text-sm">{t.noAnswer}</span>
          </div>
        )}
        {isRetrievalOnly && message.sources && message.sources.length > 0 && (
          <p className="text-sm text-gray-500 dark:text-gray-400 italic mb-2">
            {t.retrievalOnly}
          </p>
        )}
        {message.content && (
          <div className="text-sm leading-relaxed whitespace-pre-wrap">
            {renderContent(message.content, setHighlightedSource)}
          </div>
        )}
        {message.timing && (
          <div className="flex items-center gap-1 mt-2 text-xs text-gray-400 dark:text-gray-500">
            <Clock size={11} />
            <span>{t.elapsed}: {message.timing.total_s.toFixed(2)}s</span>
          </div>
        )}
        {message.sources && message.sources.length > 0 && (
          <SourceList sources={message.sources} highlightedId={highlightedSource} />
        )}
      </div>
    </div>
  );
}
