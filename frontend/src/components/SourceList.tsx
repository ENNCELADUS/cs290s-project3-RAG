import { useCallback, useRef, useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import type { Source } from "../types";
import { useLabels } from "../i18n/labels";
import { SourceCard } from "./SourceCard";

interface Props {
  sources: Source[];
  highlightedId?: number | null;
}

export function SourceList({ sources, highlightedId }: Props) {
  const [expanded, setExpanded] = useState(false);
  const t = useLabels();
  const cardRefs = useRef<Map<number, HTMLDivElement>>(new Map());

  const setRef = useCallback(
    (id: number) => (el: HTMLDivElement | null) => {
      if (el) cardRefs.current.set(id, el);
      else cardRefs.current.delete(id);
    },
    [],
  );

  if (!sources.length) return null;

  const visibleSources = expanded ? sources : sources.slice(0, 3);

  return (
    <div className="mt-2 space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
          {t.sources} ({sources.length})
        </span>
        {sources.length > 3 && (
          <button
            onClick={() => setExpanded(!expanded)}
            className="text-xs text-blue-600 dark:text-blue-400 hover:underline flex items-center gap-0.5"
          >
            {expanded ? (
              <>
                <ChevronUp size={12} /> {t.collapseSources}
              </>
            ) : (
              <>
                <ChevronDown size={12} /> {t.expandSources}
              </>
            )}
          </button>
        )}
      </div>
      <div className="space-y-1.5">
        {visibleSources.map((source) => (
          <SourceCard
            key={source.source_id}
            source={source}
            highlighted={highlightedId === source.source_id}
            onRef={setRef(source.source_id)}
          />
        ))}
      </div>
    </div>
  );
}
