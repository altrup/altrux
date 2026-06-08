import { useEffect, useRef, type KeyboardEvent } from "react";
import { LuArrowUp } from "react-icons/lu";

function SubmitButton({ disabled = false }: { disabled: boolean }) {
  return (
    <button
      type="submit"
      disabled={disabled}
      aria-label="Send message"
      className="m-2 shrink-0 size-8 rounded-xl flex items-center justify-center bg-accent text-on-accent transition-colors hover:bg-accent-hover disabled:opacity-40 disabled:cursor-not-allowed cursor-pointer"
    >
      <LuArrowUp size={14} />
    </button>
  );
}

interface ChatInputProps {
  value: string;
  onChange: (value: string) => void;
  onSend: () => void;
  disabled?: boolean;
  placeholder?: string;
  focusKey?: number;
  weight?: number | "";
  onWeightChange?: (weight: number | "") => void;
}

export default function ChatInput({
  value,
  onChange,
  onSend,
  disabled = false,
  placeholder = "Type a message…",
  focusKey = 0,
  weight,
  onWeightChange,
}: ChatInputProps) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const formRef = useRef<HTMLFormElement>(null);

  useEffect(() => {
    if (focusKey > 0) textareaRef.current?.focus();
  }, [focusKey]);

  const showWeight = weight !== undefined && onWeightChange !== undefined;

  function handleSubmit(e: { preventDefault(): void }) {
    e.preventDefault();
    if (!disabled && value.trim() && !(showWeight && weight === "")) onSend();
  }

  function handleKeyDown(e: KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      formRef.current?.requestSubmit();
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
    <form
      ref={formRef}
      onSubmit={handleSubmit}
      className="group flex-1 flex items-end bg-surface-raised rounded-2xl border border-border hover:border-border-strong focus-within:border-border-strong transition-colors"
    >
      <label className="flex-1 cursor-text self-stretch flex items-end">
        <textarea
          ref={textareaRef}
          rows={1}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          onKeyDown={handleKeyDown}
          onInput={handleInput}
          placeholder={placeholder}
          className="w-full p-3 resize-none bg-transparent text-text placeholder:text-text-faint text-sm leading-relaxed outline-none max-h-[200px]"
        />
        {!showWeight && (
          <SubmitButton
            disabled={
              disabled || !value.trim() || (showWeight && weight === "")
            }
          />
        )}
      </label>
      {showWeight && (
        <label className="cursor-text self-stretch flex items-end">
          <div className="self-stretch w-[0.5px] bg-border group-hover:bg-border-strong group-focus-within:bg-border-strong transition-colors" />
          <input
            type="number"
            min={0}
            max={1}
            step={0.1}
            value={weight}
            placeholder="Weight"
            onChange={(e) => {
              const raw = e.target.value;
              if (raw === "") {
                onWeightChange("");
              } else {
                const v = parseFloat(raw);
                if (!isNaN(v)) onWeightChange(Math.min(1, Math.max(0, v)));
              }
            }}
            className="w-24 p-3 resize-none bg-transparent text-text placeholder:text-text-faint text-sm leading-relaxed outline-none [appearance:textfield] [&::-webkit-outer-spin-button]:appearance-none [&::-webkit-inner-spin-button]:appearance-none"
          />
          <SubmitButton
            disabled={
              disabled || !value.trim() || (showWeight && weight === "")
            }
          />
        </label>
      )}
    </form>
  );
}
