interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
}

export default function ChatMessage({
  role,
  content,
  isStreaming = false,
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

  return (
    <div className="group relative flex flex-col gap-1 max-w-[85%]">
      <div className="relative text-text text-sm w-fit leading-relaxed whitespace-pre-wrap">
        {content}
        {isStreaming && (
          <span className="inline-block w-0.5 h-4 ml-0.5 bg-text-muted align-middle animate-pulse" />
        )}
      </div>
    </div>
  );
}
