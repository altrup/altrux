import { useEffect, useRef, useState } from "react";
import type { Route } from "./+types/home";
import ChatInput from "~/components/ChatInput";
import ChatMessage from "~/components/ChatMessage";
import HealthBadge from "~/components/HealthBadge";
import ThemeToggle from "~/components/ThemeToggle";
import {
  addUserMessage,
  getSession,
  resetSession,
  streamGenerate,
  type Message,
} from "~/lib/api";

// eslint-disable-next-line no-empty-pattern
export function meta({}: Route.MetaArgs) {
  return [{ title: "continual-learning" }];
}

export default function Home() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getSession().then(setMessages).catch(() => {});
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleSend() {
    const content = input.trim();
    if (!content || isGenerating) return;

    setInput("");
    setMessages((prev) => [...prev, { role: "user", content }]);
    setIsGenerating(true);

    try {
      await addUserMessage(content);
      setMessages((prev) => [...prev, { role: "assistant", content: "" }]);

      for await (const token of streamGenerate()) {
        setMessages((prev) => {
          const updated = [...prev];
          const last = updated[updated.length - 1];
          updated[updated.length - 1] = {
            ...last,
            content: last.content + token.token,
          };
          return updated;
        });
      }
    } catch (err) {
      setMessages((prev) => {
        const updated = [...prev];
        const last = updated[updated.length - 1];
        if (last?.role === "assistant" && last.content === "") {
          updated[updated.length - 1] = {
            role: "assistant",
            content: `Error: ${err instanceof Error ? err.message : "Something went wrong"}`,
          };
        }
        return updated;
      });
    } finally {
      setIsGenerating(false);
    }
  }

  async function handleReset() {
    if (isGenerating) return;
    try {
      await resetSession();
      setMessages([]);
    } catch {
      // silently ignore — local state is still cleared
    }
  }

  return (
    <div className="flex flex-col h-screen bg-page">
      {/* Header */}
      <header className="shrink-0 flex items-center justify-between px-6 py-3 bg-surface border-b border-border">
        <span className="text-sm font-semibold text-text tracking-tight">
          continual-learning
        </span>
        <div className="flex items-center gap-3">
          <HealthBadge />
          <div className="w-px h-4 bg-border" />
          <button
            onClick={handleReset}
            disabled={isGenerating}
            className="text-sm text-text-muted hover:text-text transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
          >
            Reset
          </button>
          <ThemeToggle />
        </div>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto px-4 py-6">
        <div className="mx-auto max-w-2xl flex flex-col gap-6">
          {messages.length === 0 && (
            <div className="flex items-center justify-center min-h-[40vh]">
              <p className="text-text-muted text-sm">
                Start a conversation below.
              </p>
            </div>
          )}
          {messages.map((msg, i) => (
            <ChatMessage
              key={i}
              role={msg.role}
              content={msg.content}
              isStreaming={
                isGenerating &&
                i === messages.length - 1 &&
                msg.role === "assistant"
              }
            />
          ))}
          <div ref={bottomRef} />
        </div>
      </main>

      {/* Input */}
      <footer className="shrink-0 px-4 pb-6 pt-2">
        <div className="mx-auto max-w-2xl">
          <ChatInput
            value={input}
            onChange={setInput}
            onSend={handleSend}
            disabled={isGenerating}
          />
        </div>
      </footer>
    </div>
  );
}
