import { useEffect } from "react";
import { useChat } from "./hooks/useChat";
import { fetchHealth, fetchSamples } from "./api/client";
import { ChatWindow } from "./components/ChatWindow";
import { InputBar } from "./components/InputBar";
import { SampleQuestions } from "./components/SampleQuestions";
import { SettingsPanel } from "./components/SettingsPanel";
import { StatusBadge } from "./components/StatusBadge";
import { useLabels } from "./i18n/labels";

const DEFAULT_SAMPLES = [
  "上海科技大学一共有几个学院？",
  "《深度学习》这门课的任课老师是谁？",
  "计算机科学与技术专业需要修满多少学分才能毕业？",
  "信息学院专业型硕士与学术型硕士的培养方案有什么不同？",
  "我想做机器人方向，有哪些导师可以推荐？",
  "Which SIST faculty work on robotics?",
];

export default function App() {
  const { state, send, updateSettings, setStatus, setSamples } = useChat();
  const t = useLabels();

  useEffect(() => {
    fetchHealth()
      .then((health) => setStatus(health))
      .catch(() => setStatus({ status: "error", mode: "unavailable", artifacts_loaded: false, generator_loaded: false }));

    fetchSamples()
      .then((resp) => setSamples(resp.questions))
      .catch(() => setSamples(DEFAULT_SAMPLES));
  }, []);

  const hasMessages = state.messages.length > 0;
  const generatorAvailable = state.systemStatus?.generator_loaded ?? false;

  return (
    <div className="flex flex-col h-screen bg-white dark:bg-gray-900">
      {/* Header */}
      <header className="shrink-0 flex items-center justify-between px-4 py-3 border-b border-gray-200 dark:border-gray-700 bg-white/80 dark:bg-gray-900/80 backdrop-blur-sm">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-lg bg-blue-600 flex items-center justify-center">
            <span className="text-white font-bold text-sm">R</span>
          </div>
          <div>
            <h1 className="text-sm font-semibold text-gray-800 dark:text-gray-100">{t.title}</h1>
            <p className="text-xs text-gray-500 dark:text-gray-400">{t.subtitle}</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <StatusBadge status={state.systemStatus} />
          <SettingsPanel
            settings={state.settings}
            onUpdate={updateSettings}
            generatorAvailable={generatorAvailable}
          />
        </div>
      </header>

      {/* Content */}
      {hasMessages ? (
        <ChatWindow messages={state.messages} isLoading={state.isLoading} />
      ) : (
        <div className="flex-1 overflow-y-auto">
          <SampleQuestions
            questions={state.samples}
            onSelect={(q) => send(q)}
          />
        </div>
      )}

      {/* Input */}
      <InputBar onSend={send} disabled={state.isLoading} />
    </div>
  );
}
