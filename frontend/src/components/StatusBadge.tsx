import type { HealthResponse } from "../types";
import { useLabels } from "../i18n/labels";

interface Props {
  status: HealthResponse | null;
}

export function StatusBadge({ status }: Props) {
  const t = useLabels();

  if (!status) {
    return (
      <div className="flex items-center gap-2 text-sm text-gray-500">
        <span className="w-2 h-2 rounded-full bg-gray-400 pulse-dot" />
        {t.statusConnecting}
      </div>
    );
  }

  if (status.mode === "unavailable" || status.status === "error") {
    return (
      <div className="flex items-center gap-2 text-sm text-red-600 dark:text-red-400">
        <span className="w-2 h-2 rounded-full bg-red-500" />
        {t.statusUnavailable}
      </div>
    );
  }

  const isFullRag = status.generator_loaded;
  return (
    <div className={`flex items-center gap-2 text-sm ${isFullRag ? "text-green-600 dark:text-green-400" : "text-yellow-600 dark:text-yellow-400"}`}>
      <span className={`w-2 h-2 rounded-full ${isFullRag ? "bg-green-500" : "bg-yellow-500"}`} />
      {isFullRag ? t.statusFullRag : t.statusRetrievalOnly}
    </div>
  );
}
