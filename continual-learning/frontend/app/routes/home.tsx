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

interface ReviseTarget {
  n: number; // N model messages back from atTurn — goes inside <revise back=N>
  atTurn: number; // message index of the last assistant turn this revision is anchored to
  isEdit: boolean;
}

export default function Home() {
  const { online } = useHealth();
  const [messages, setMessages] = useState<Message[]>([]);
  // keyed by atTurn so revisions stay on the right message as the conversation grows
  const [reviseByIndex, setReviseByIndex] = useState<
    Record<number, ReviseEntry[]>
  >({});
  const [input, setInput] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [reviseTarget, setReviseTarget] = useState<ReviseTarget | null>(null);
  const [revisionInput, setRevisionInput] = useState("");
  const [revisionWeight, setRevisionWeight] = useState<number | "">("");
  const [focusKey, setFocusKey] = useState(0);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    getSession()
      .then(({ messages, reviseSuggestions }) => {
        setMessages(messages);
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
    setReviseTarget(null);
    setRevisionInput("");
    setRevisionWeight("");
    setReviseByIndex({});
    try {
      await resetSession();
      setMessages([]);
    } catch {
      // silently ignore — local state is still cleared
    }
  }

  async function handleRevise() {
    if (!reviseTarget || !revisionInput.trim()) return;
    const { n, atTurn } = reviseTarget;
    const trimmed = revisionInput.trim();
    const weight = revisionWeight === "" ? undefined : revisionWeight;
    const tag = `<revise back=${n}>${trimmed}</revise weight=${weight ?? 0.5}>`;
    try {
      await submitRevision(n, trimmed, weight, atTurn);
      setReviseByIndex((prev) => {
        const filtered = (prev[atTurn] ?? []).filter(
          (e) => getBackValue(e.revision) !== n,
        );
        return {
          ...prev,
          [atTurn]: sortByBackDesc([...filtered, { atTurn, revision: tag }]),
        };
      });
    } catch {
      // silently ignore — revision may have failed but don't block the UI
    }
    setRevisionInput("");
    setRevisionWeight("");
    setReviseTarget(null);
    setFocusKey((k) => k + 1);
  }

  function handleEditRevision(entry: ReviseEntry) {
    const backVal = getBackValue(entry.revision);
    const textMatch = entry.revision.match(/back=\d+>([\s\S]*?)<\/revise/);
    const weightMatch = entry.revision.match(/revise weight=([\d.]+)>/);
    setReviseTarget({
      n: backVal,
      atTurn: entry.atTurn,
      isEdit: true,
    });
    setRevisionInput(textMatch ? textMatch[1] : "");
    setRevisionWeight(weightMatch ? parseFloat(weightMatch[1]) : "");
    setFocusKey((k) => k + 1);
  }

  function handleReviseKeyDown(e: React.KeyboardEvent) {
    if (e.key === "Escape") {
      setReviseTarget(null);
      setRevisionInput("");
      setRevisionWeight("");
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
            const iRank = assistantIndices.indexOf(i);
            const assistantRankFromEnd =
              msg.role === "assistant"
                ? assistantIndices.length - 1 - iRank
                : 0;

            // Whether the current last turn already has a revision for this message
            const hasRevisionForN = (
              reviseByIndex[lastAssistantIndex] ?? []
            ).some((e) => getBackValue(e.revision) === assistantRankFromEnd);
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
                canRevise={
                  msg.role === "assistant" && !isLastAsst && !isGenerating
                }
                isLastAssistant={isLastAsst}
                isReviseTarget={
                  reviseTarget !== null &&
                  assistantIndices[
                    assistantIndices.indexOf(reviseTarget.atTurn) -
                      reviseTarget.n
                  ] === i
                }
                reviseSuggestions={reviseByIndex[i] ?? []}
                revisionLabel={
                  hasRevisionForN ? "Edit revision" : "Add revision"
                }
                onRevise={() => {
                  if (hasRevisionForN) {
                    const existing = (
                      reviseByIndex[lastAssistantIndex] ?? []
                    ).find(
                      (e) => getBackValue(e.revision) === assistantRankFromEnd,
                    )!;
                    handleEditRevision(existing);
                  } else {
                    setReviseTarget({
                      n: assistantRankFromEnd,
                      atTurn: lastAssistantIndex,
                      isEdit: false,
                    });
                    setRevisionInput("");
                    setRevisionWeight("");
                    setFocusKey((k) => k + 1);
                  }
                }}
                onEdit={(entry) => handleEditRevision(entry)}
              />
            );
          })}
          <div ref={bottomRef} />
        </div>

        <div className="sticky bottom-0 px-2 pb-6 bg-page">
          <div className="mx-auto max-w-[45rem]">
            {reviseTarget !== null && (
              <div className="flex items-center justify-between px-3 py-1.5 mb-2 rounded-xl bg-surface border border-border text-xs text-text-muted">
                <span>
                  {reviseTarget.isEdit ? "Editing revision" : "Adding revision"}{" for assistant message "}
                  ({reviseTarget.n} {reviseTarget.n === 1 ? "turn" : "turns"}{" "}
                  back)
                </span>
                <button
                  onClick={() => {
                    setReviseTarget(null);
                    setRevisionInput("");
                    setRevisionWeight("");
                    setFocusKey((k) => k + 1);
                  }}
                  className="ml-3 text-text-faint hover:text-text-muted transition-colors cursor-pointer"
                >
                  ✕
                </button>
              </div>
            )}
            <div onKeyDown={reviseTarget ? handleReviseKeyDown : undefined}>
              <ChatInput
                value={reviseTarget !== null ? revisionInput : input}
                onChange={reviseTarget !== null ? setRevisionInput : setInput}
                onSend={reviseTarget !== null ? handleRevise : handleSend}
                disabled={isGenerating || !online}
                placeholder={
                  reviseTarget !== null
                    ? "Type a revision…"
                    : !online
                      ? "Backend offline…"
                      : "Type a message…"
                }
                focusKey={focusKey}
                weight={reviseTarget !== null ? revisionWeight : undefined}
                onWeightChange={
                  reviseTarget !== null ? setRevisionWeight : undefined
                }
              />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
