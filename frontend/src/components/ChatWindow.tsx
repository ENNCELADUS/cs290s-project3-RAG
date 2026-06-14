import { useEffect, useRef } from "react";
import { Loader2 } from "lucide-react";
import type { Message } from "../types";
import { useLabels } from "../i18n/labels";
import { ChatMessage } from "./ChatMessage";

interface Props {
  messages: Message[];
  isLoading: boolean;
}

export function ChatWindow({ messages, isLoading }: Props) {
  const t = useLabels();
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  return (
    <div className="flex-1 overflow-y-auto px-4 py-6 space-y-4">
      {messages.map((msg) => (
        <ChatMessage key={msg.id} message={msg} />
      ))}
      {isLoading && (
        <div className="flex justify-start fade-in">
          <div className="chat-bubble-assistant">
            <div className="flex items-center gap-2 text-sm text-gray-500 dark:text-gray-400">
              <Loader2 size={16} className="animate-spin" />
              <span>{t.loading}</span>
            </div>
          </div>
        </div>
      )}
      <div ref={bottomRef} />
    </div>
  );
}
