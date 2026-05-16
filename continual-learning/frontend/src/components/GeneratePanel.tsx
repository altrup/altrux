import { useEffect, useRef, useState } from "react";
import type { GenerateStatus } from "../types";

interface Message {
  role: "user" | "assistant";
  text: string;
}

interface Props {
  response: string;
  reward: number | null;
  status: GenerateStatus;
  prefillProgress: number; // 0–1, driven by real backend layer events
  promptTokens: string[];  // token strings for the committed prompt, from backend tokenizer
  onGenerate: (prompt: string, maxNewTokens: number) => void;
  onAbort: () => void;
}

export function GeneratePanel({
  response,
  reward,
  status,
  prefillProgress,
  promptTokens,
  onGenerate,
  onAbort,
}: Props) {
  const [inputText, setInputText] = useState("");
  const [maxTokens, setMaxTokens] = useState(256);
  const [committedPrompt, setCommittedPrompt] = useState("");
  const [history, setHistory] = useState<Message[]>([]);

  const chatBottomRef = useRef<HTMLDivElement>(null);
  const isBusy = status === "prefill" || status === "streaming";

  // Reveal tokens one at a time driven by backend layer-progress events.
  // Falls back to the raw prompt string when tokens aren't available yet.
  const visibleText =
    status === "prefill" && promptTokens.length > 0
      ? promptTokens
          .slice(0, Math.floor(prefillProgress * promptTokens.length))
          .join("")
      : committedPrompt;

  const showCursor = status === "prefill" && visibleText.length < committedPrompt.length;
  const showAssistant = status === "streaming" || status === "done";

  // Auto-scroll as content arrives
  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [response, visibleText]);

  const handleSend = () => {
    const text = inputText.trim();
    if (!text || isBusy) return;

    // Archive completed exchange into history before starting a new one
    if (committedPrompt && status === "done") {
      setHistory((h) => [
        ...h,
        { role: "user", text: committedPrompt },
        { role: "assistant", text: response },
      ]);
    }

    setCommittedPrompt(text);
    setInputText("");
    onGenerate(text, maxTokens);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div style={{ display: "flex", flexDirection: "column", flex: 1, minHeight: 0 }}>
      {/* ── chat history ── */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          padding: "1rem",
          display: "flex",
          flexDirection: "column",
          gap: "0.75rem",
        }}
      >
        {history.map((msg, i) => (
          <div
            key={i}
            style={{ display: "flex", justifyContent: msg.role === "user" ? "flex-end" : "flex-start" }}
          >
            <div style={msg.role === "user" ? userBubbleStyle : assistantBubbleStyle}>
              {msg.text}
            </div>
          </div>
        ))}

        {/* Current user bubble — reveals in sync with backend layer progress */}
        {committedPrompt && status !== "idle" && (
          <div style={{ display: "flex", justifyContent: "flex-end" }}>
            <div style={userBubbleStyle}>
              {visibleText}
              {showCursor && <span style={{ opacity: 0.5 }}>▋</span>}
            </div>
          </div>
        )}

        {/* "Processing…" shown after typewriter finishes but before first token */}
        {status === "prefill" && prefillProgress >= 1 && (
          <div style={{ color: "#999", fontSize: "0.82rem", paddingLeft: "0.25rem" }}>
            Processing…
          </div>
        )}

        {/* Assistant bubble */}
        {showAssistant && (
          <div style={{ display: "flex", justifyContent: "flex-start" }}>
            <div style={assistantBubbleStyle}>
              {response}
              {status === "streaming" && <span style={{ opacity: 0.5 }}>▋</span>}
            </div>
          </div>
        )}

        {status === "done" && reward !== null && (
          <div style={{ fontSize: "0.8rem", color: "#999", paddingLeft: "0.25rem" }}>
            Critic reward: {reward.toFixed(3)}
          </div>
        )}

        <div ref={chatBottomRef} />
      </div>

      {/* ── input bar ── */}
      <div
        style={{
          borderTop: "1px solid #e0e0e0",
          padding: "0.75rem 1rem",
          display: "flex",
          flexDirection: "column",
          gap: "0.4rem",
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.5rem", fontSize: "0.78rem", color: "#888" }}>
          <span>Max tokens: {maxTokens}</span>
          <input
            type="range"
            min={16}
            max={512}
            value={maxTokens}
            onChange={(e) => setMaxTokens(Number(e.target.value))}
            style={{ flex: 1 }}
          />
        </div>

        <div style={{ display: "flex", gap: "0.5rem", alignItems: "flex-end" }}>
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Type a message… (Enter to send, Shift+Enter for newline)"
            rows={2}
            disabled={isBusy}
            style={{
              flex: 1,
              resize: "none",
              padding: "0.5rem 0.75rem",
              borderRadius: "12px",
              border: "1px solid #ccc",
              fontFamily: "inherit",
              fontSize: "0.95rem",
              lineHeight: 1.4,
            }}
          />
          {isBusy ? (
            <button
              onClick={onAbort}
              style={{ background: "#fee", border: "1px solid #fcc", borderRadius: "8px", padding: "0.5rem 1rem" }}
            >
              Stop
            </button>
          ) : (
            <button
              onClick={handleSend}
              disabled={!inputText.trim()}
              style={{ borderRadius: "8px", padding: "0.5rem 1rem" }}
            >
              Send
            </button>
          )}
        </div>
      </div>
    </div>
  );
}

const userBubbleStyle: React.CSSProperties = {
  background: "#007aff",
  color: "#fff",
  padding: "0.6rem 1rem",
  borderRadius: "18px 18px 4px 18px",
  maxWidth: "75%",
  whiteSpace: "pre-wrap",
  lineHeight: 1.5,
};

const assistantBubbleStyle: React.CSSProperties = {
  background: "#f0f0f0",
  color: "#111",
  padding: "0.6rem 1rem",
  borderRadius: "18px 18px 18px 4px",
  maxWidth: "75%",
  whiteSpace: "pre-wrap",
  fontFamily: "monospace",
  fontSize: "0.9rem",
  lineHeight: 1.5,
};
