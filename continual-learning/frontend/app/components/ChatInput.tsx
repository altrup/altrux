import { useEffect, useRef, type KeyboardEvent } from "react";
import { LuArrowUp } from "react-icons/lu";

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  disabled?: boolean;
  placeholder?: string;
  focusKey?: number;
}

export default function ChatInput({
  value,
  onChange,
  onSend,
  disabled = false,
  placeholder = "Type a message…",
  focusKey = 0,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    if (focusKey > 0) textareaRef.current?.focus();
  }, [focusKey]);

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      if (!disabled && value.trim()) onSend();
    }
  }

  useEffect(() => {
    const el = textareaRef.current;
    if (!el || value !== "") return;
    el.style.height = "auto";
  }, [value]);

  function handleInput() {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(el.scrollHeight, 200)}px`;
  }

  return (
    <label className="flex items-end gap-2 bg-surface-raised rounded-2xl border border-border hover:border-border-strong focus-within:border-border-strong transition-colors p-2 cursor-text">
      <textarea
        ref={textareaRef}
        rows={1}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={handleKeyDown}
        onInput={handleInput}
        placeholder={placeholder}
        className="flex-1 resize-none bg-transparent text-text placeholder:text-text-faint text-sm leading-relaxed outline-none min-h-8 max-h-[200px] px-2 py-1.5"
      />
      <button
        onClick={onSend}
        disabled={disabled || !value.trim()}
        aria-label="Send message"
        className="shrink-0 size-8 rounded-xl flex items-center justify-center bg-accent text-on-accent transition-colors hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
      >
        <LuArrowUp size={14} />
      </button>
    </label>
  );
}
