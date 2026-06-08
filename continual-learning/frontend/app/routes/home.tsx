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
  submitRevision,
  type Message,
  type ReviseEntry,
} from "~/lib/api";
import { useHealth } from "~/lib/useHealth";

// eslint-disable-next-line no-empty-pattern
export function meta({}: Route.MetaArgs) {
  return [{ title: "continual-learning" }];
}

function getBackValue(revision: string): number {
  const m = revision.match(/back=(\d+)/);
  return m ? parseInt(m[1], 10) : 0;
}

function sortByBackDesc(entries: ReviseEntry[]): ReviseEntry[] {
  return [...entries].sort(
    (a, b) => getBackValue(b.revision) - getBackValue(a.revision),
  );
}

interface SuggestTarget {
  messageIndex: number;
  n: number; // N model messages back from last — goes inside <revise turn=N>
}

export default function Home() {
  const { online } = useHealth();
  const [messages, setMessages] = useState<Message[]>([]);
  // keyed by message index so suggestions stay on the right message as the conversation grows
  const [reviseByIndex, setReviseByIndex] = useState<
    Record<number, ReviseEntry[]>
  >({});
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [suggestTarget, setSuggestTarget] = useState<SuggestTarget | null>(
    null,
  );
  const [suggestInput, setSuggestInput] = useState("");
  const [suggestWeight, setSuggestWeight] = useState<number | "">("");
  const [focusKey, setFocusKey] = useState(0);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getSession()
      .then(({ messages, reviseSuggestions }) => {
        setMessages(messages);
        // Map flat revise list back to per-message indices.
        // Parse N from "<revise turn=N>..." in each suggestion, then walk N assistant
        // messages back from atTurn to find the target message index.
        const byIndex: Record<number, ReviseEntry[]> = {};
        for (const entry of reviseSuggestions) {
          byIndex[entry.atTurn] = [...(byIndex[entry.atTurn] ?? []), entry];
        }
        for (const key of Object.keys(byIndex)) {
          byIndex[Number(key)] = sortByBackDesc(byIndex[Number(key)]);
        }
        setReviseByIndex(byIndex);
      })
      .catch(() => {});
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
    setSuggestTarget(null);
    setSuggestInput("");
    setSuggestWeight("");
    setReviseByIndex({});
    try {
      await resetSession();
      setMessages([]);
    } catch {
      // silently ignore — local state is still cleared
    }
  }

  async function handleSuggest() {
    if (!suggestTarget || !suggestInput.trim()) return;
    const { n } = suggestTarget;
    const trimmed = suggestInput.trim();
    const weight = suggestWeight === "" ? undefined : suggestWeight;
    const tag = `<revise back=${n}>${trimmed}</revise weight=${weight ?? 0.5}>`;
    const atTurn = assistantIndices[assistantIndices.length - 1];
    try {
      await submitRevision(n, trimmed, weight);
      setReviseByIndex((prev) => ({
        ...prev,
        [atTurn]: sortByBackDesc([
          ...(prev[atTurn] ?? []),
          { atTurn, revision: tag },
        ]),
      }));
    } catch {
      // silently ignore — revision may have failed but don't block the UI
    }
    setSuggestInput("");
    setSuggestWeight("");
    setSuggestTarget(null);
    setFocusKey((k) => k + 1);
  }

  function handleSuggestKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Escape") {
      setSuggestTarget(null);
      setSuggestInput("");
      setSuggestWeight("");
      setFocusKey((k) => k + 1);
    }
  }

  // Precompute assistant message indices for turn calculation.
  const assistantIndices = messages
    .map((m, i) => (m.role === "assistant" ? i : -1))
    .filter((i) => i !== -1);
  const lastAssistantIndex =
    assistantIndices.length > 0
      ? assistantIndices[assistantIndices.length - 1]
      : -1;

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
            className="text-sm text-text-muted hover:text-error transition-colors disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
          >
            Reset
          </button>
          <ThemeToggle />
        </div>
      </header>

      {/* Messages + sticky input */}
      <main className="flex-1 overflow-y-scroll flex flex-col px-4 pt-12">
        <div className="mx-auto w-full max-w-2xl flex-grow flex flex-col gap-6 pb-12">
          {messages.length === 0 && (
            <div className="flex items-center justify-center min-h-[40vh]">
              <p className="text-text-muted text-sm">
                Start a conversation below.
              </p>
            </div>
          )}
          {messages.map((msg, i) => {
            const isLastAsst = i === lastAssistantIndex;
            // turn = number of assistant messages after this one (inclusive of this one would be 0, so we count strictly after)
            const assistantRankFromEnd =
              msg.role === "assistant"
                ? assistantIndices.length - 1 - assistantIndices.indexOf(i)
                : 0;
            return (
              <ChatMessage
                key={i}
                role={msg.role}
                content={msg.content}
                isStreaming={
                  isGenerating &&
                  i === messages.length - 1 &&
                  msg.role === "assistant"
                }
                canSuggest={
                  msg.role === "assistant" && !isLastAsst && !isGenerating
                }
                isLastAssistant={isLastAsst}
                reviseSuggestions={reviseByIndex[i] ?? []}
                onSuggest={() => {
                  setSuggestTarget({
                    messageIndex: i,
                    n: assistantRankFromEnd,
                  });
                  setFocusKey((k) => k + 1);
                }}
              />
            );
          })}
          <div ref={bottomRef} />
        </div>

        <div className="sticky bottom-0 px-2 pb-6 bg-page">
          <div className="mx-auto max-w-[45rem]">
            {suggestTarget !== null && (
              <div className="flex items-center justify-between px-3 py-1.5 mb-2 rounded-xl bg-surface border border-border text-xs text-text-muted">
                <span>
                  Suggesting for assistant message ({suggestTarget.n}{" "}
                  {suggestTarget.n === 1 ? "turn" : "turns"} back)
                </span>
                <button
                  onClick={() => {
                    setSuggestTarget(null);
                    setSuggestInput("");
                    setSuggestWeight("");
                    setFocusKey((k) => k + 1);
                  }}
                  className="ml-3 text-text-faint hover:text-text-muted transition-colors cursor-pointer"
                >
                  ✕
                </button>
              </div>
            )}
            <div onKeyDown={suggestTarget ? handleSuggestKeyDown : undefined}>
              <ChatInput
                value={suggestTarget !== null ? suggestInput : input}
                onChange={suggestTarget !== null ? setSuggestInput : setInput}
                onSend={suggestTarget !== null ? handleSuggest : handleSend}
                disabled={isGenerating || !online}
                placeholder={
                  suggestTarget !== null
                    ? "Type a better response…"
                    : !online
                      ? "Backend offline…"
                      : "Type a message…"
                }
                focusKey={focusKey}
                weight={suggestTarget !== null ? suggestWeight : undefined}
                onWeightChange={
                  suggestTarget !== null ? setSuggestWeight : undefined
                }
              />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
