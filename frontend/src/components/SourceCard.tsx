import { useState } from "react";
import { ChevronDown, ChevronRight, ExternalLink } from "lucide-react";
import type { Source } from "../types";
import clsx from "clsx";

interface Props {
  source: Source;
  highlighted?: boolean;
  onRef?: (el: HTMLDivElement | null) => void;
}

export function SourceCard({ source, highlighted, onRef }: Props) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div
      ref={onRef}
      className={clsx("source-card", highlighted && "highlighted")}
      onClick={() => setExpanded(!expanded)}
    >
      <div className="flex items-start gap-2">
        <span className="citation-badge mt-0.5">{source.source_id}</span>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            {expanded ? <ChevronDown size={14} /> : <ChevronRight size={14} />}
            <span className="font-medium text-sm truncate">
              {source.title || "(untitled)"}
            </span>
            {source.score !== null && (
              <span className="text-xs text-gray-500 dark:text-gray-400 ml-auto shrink-0">
                {source.score.toFixed(4)}
              </span>
            )}
          </div>
          {expanded && (
            <div className="mt-2 space-y-1.5 fade-in">
              <p className="text-sm text-gray-600 dark:text-gray-300 leading-relaxed">
                {source.snippet}
              </p>
              {source.url && (
                <a
                  href={source.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="inline-flex items-center gap-1 text-xs text-blue-600 dark:text-blue-400 hover:underline"
                  onClick={(e) => e.stopPropagation()}
                >
                  <ExternalLink size={12} />
                  {source.url.length > 60 ? source.url.slice(0, 60) + "..." : source.url}
                </a>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
