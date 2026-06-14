import { Settings, X } from "lucide-react";
import { useState } from "react";
import type { ChatSettings, RetrievalMode } from "../types";
import { useLabels } from "../i18n/labels";

interface Props {
  settings: ChatSettings;
  onUpdate: (patch: Partial<ChatSettings>) => void;
  generatorAvailable: boolean;
}

const MODES: { value: RetrievalMode; labelKey: "modeHybrid" | "modeDense" | "modeBm25" }[] = [
  { value: "hybrid", labelKey: "modeHybrid" },
  { value: "dense", labelKey: "modeDense" },
  { value: "bm25", labelKey: "modeBm25" },
];

export function SettingsPanel({ settings, onUpdate, generatorAvailable }: Props) {
  const [open, setOpen] = useState(false);
  const t = useLabels();

  return (
    <div className="relative">
      <button
        onClick={() => setOpen(!open)}
        className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors text-gray-600 dark:text-gray-400"
        title={t.settings}
      >
        <Settings size={18} />
      </button>

      {open && (
        <>
          <div className="fixed inset-0 z-10" onClick={() => setOpen(false)} />
          <div className="settings-panel absolute right-0 top-full mt-2 z-20 w-72 fade-in">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium">{t.settings}</span>
              <button onClick={() => setOpen(false)} className="p-1 rounded hover:bg-gray-100 dark:hover:bg-gray-700">
                <X size={14} />
              </button>
            </div>

            <div className="space-y-4">
              <div>
                <label className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                  {t.retrievalMode}
                </label>
                <div className="mt-1.5 flex gap-1">
                  {MODES.map(({ value, labelKey }) => (
                    <button
                      key={value}
                      onClick={() => onUpdate({ mode: value })}
                      className={`flex-1 px-3 py-1.5 text-xs rounded-lg transition-colors ${
                        settings.mode === value
                          ? "bg-blue-600 text-white"
                          : "bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-200 dark:hover:bg-gray-600"
                      }`}
                    >
                      {t[labelKey]}
                    </button>
                  ))}
                </div>
              </div>

              <div>
                <label className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
                  {t.topK}: {settings.topK}
                </label>
                <input
                  type="range"
                  min={1}
                  max={10}
                  value={settings.topK}
                  onChange={(e) => onUpdate({ topK: parseInt(e.target.value, 10) })}
                  className="mt-1.5 w-full accent-blue-600"
                />
                <div className="flex justify-between text-xs text-gray-400">
                  <span>1</span>
                  <span>10</span>
                </div>
              </div>

              <div className="flex items-center justify-between">
                <div>
                  <label className="text-xs font-medium text-gray-700 dark:text-gray-300">
                    {t.retrievalOnly}
                  </label>
                  <p className="text-xs text-gray-400 mt-0.5">{t.retrievalOnlyDesc}</p>
                </div>
                <button
                  onClick={() => onUpdate({ retrievalOnly: !settings.retrievalOnly })}
                  disabled={!generatorAvailable}
                  className={`relative w-10 h-5 rounded-full transition-colors ${
                    settings.retrievalOnly ? "bg-blue-600" : "bg-gray-300 dark:bg-gray-600"
                  } ${!generatorAvailable ? "opacity-50 cursor-not-allowed" : "cursor-pointer"}`}
                >
                  <span
                    className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow transition-transform ${
                      settings.retrievalOnly ? "translate-x-5" : "translate-x-0.5"
                    }`}
                  />
                </button>
              </div>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
