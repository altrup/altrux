import type { ReviseEntry } from "~/lib/api";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
  canSuggest?: boolean;
  isLastAssistant?: boolean;
  reviseSuggestions?: ReviseEntry[];
  onSuggest?: () => void;
}

export default function ChatMessage({
  role,
  content,
  isStreaming = false,
  canSuggest = false,
  isLastAssistant = false,
  reviseSuggestions = [],
  onSuggest,
}: ChatMessageProps) {
  if (role === "user") {
    return (
      <div className="flex justify-end">
        <div className="max-w-[70%] rounded-3xl px-4 py-2.5 bg-user-bubble text-user-bubble-text text-sm leading-relaxed">
          {content}
        </div>
      </div>
    );
  }

  const suggestTitle =
    "Suggestions are only available for messages you've already replied to";

  return (
    <div className="group relative flex flex-col gap-1 max-w-[85%]">
      <div className="text-text text-sm leading-relaxed whitespace-pre-wrap">
        {content}
        {isStreaming && (
          <span className="inline-block w-0.5 h-4 ml-0.5 bg-text-muted align-middle animate-pulse" />
        )}
      </div>
      {reviseSuggestions.length > 0 && (
        <div className="flex flex-col gap-1">
          {reviseSuggestions.map((r, i) => (
            <code
              key={i}
              className="block text-xs text-text-muted font-mono bg-surface rounded px-2 py-1 whitespace-pre-wrap break-words w-fit"
            >
              {r.revision}
            </code>
          ))}
        </div>
      )}
      {canSuggest && (
        <div className="w-full h-0 overflow-visible absolute bottom-0">
          <button
            onClick={onSuggest}
            title="Suggest a better response"
            className="absolute top-0 pt-1 text-xs text-text-faint opacity-0 group-hover:opacity-100 hover:text-text-muted transition-[opacity,color] cursor-pointer"
          >
            Suggest
          </button>
        </div>
      )}
      {isLastAssistant && (
        <div className="w-full h-0 overflow-visible absolute bottom-0">
          <p className="absolute top-0 pt-1 text-xs text-text-faint opacity-0 group-hover:opacity-100 transition-opacity">
            {suggestTitle}
          </p>
        </div>
      )}
    </div>
  );
}
