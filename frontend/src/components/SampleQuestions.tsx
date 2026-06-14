import { MessageSquare } from "lucide-react";
import { useLabels } from "../i18n/labels";

interface Props {
  questions: string[];
  onSelect: (question: string) => void;
}

export function SampleQuestions({ questions, onSelect }: Props) {
  const t = useLabels();

  if (!questions.length) return null;

  return (
    <div className="flex flex-col items-center justify-center h-full px-4 py-12 fade-in">
      <div className="w-14 h-14 rounded-2xl bg-blue-100 dark:bg-blue-900/30 flex items-center justify-center mb-4">
        <MessageSquare size={28} className="text-blue-600 dark:text-blue-400" />
      </div>
      <h2 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-1">{t.title}</h2>
      <p className="text-sm text-gray-500 dark:text-gray-400 mb-6">{t.subtitle}</p>
      <div className="w-full max-w-2xl">
        <p className="text-xs font-medium text-gray-400 dark:text-gray-500 uppercase tracking-wide mb-3">
          {t.sampleTitle}
        </p>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
          {questions.map((q, i) => (
            <button
              key={i}
              onClick={() => onSelect(q)}
              className="text-left px-4 py-3 rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 hover:border-blue-300 dark:hover:border-blue-600 hover:bg-blue-50 dark:hover:bg-blue-900/10 transition-colors text-sm text-gray-700 dark:text-gray-300"
            >
              {q}
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
