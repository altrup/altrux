import { AnimatePresence, motion } from "motion/react";
import type { ReviseEntry } from "~/lib/api";

interface ChatMessageProps {
  role: "user" | "assistant";
  content: string;
  isStreaming?: boolean;
  canRevise?: boolean;
  isLastAssistant?: boolean;
  isReviseTarget?: boolean;
  reviseSuggestions?: ReviseEntry[];
  revisionLabel?: string;
  onRevise?: () => void;
  onEdit?: (entry: ReviseEntry) => void;
  onDelete?: (entry: ReviseEntry) => void;
}

export default function ChatMessage({
  role,
  content,
  isStreaming = false,
  canRevise = false,
  isLastAssistant = false,
  isReviseTarget = false,
  reviseSuggestions = [],
  revisionLabel = "Add revision",
  onRevise,
  onEdit,
  onDelete,
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

  const revisionOnlyTitle =
    "Revisions are only available for messages you've already replied to";

  return (
    <div className="group relative flex flex-col gap-1 max-w-[85%]">
      <div className="group/msg relative">
        {(isLastAssistant || canRevise || isReviseTarget) && (
          <div className="w-full h-0 z-10 overflow-visible relative">
            {!isReviseTarget && isLastAssistant && (
              <p className="absolute top-0 -translate-y-full text-xs text-text-faint opacity-0 group-hover/msg:opacity-100 transition-opacity">
                {revisionOnlyTitle}
              </p>
            )}
            {!isReviseTarget && canRevise && (
              <button
                onClick={onRevise}
                title={
                  revisionLabel === "Add revision"
                    ? "Add a revision"
                    : "Edit this revision"
                }
                className="absolute top-0 -translate-y-full text-xs text-text-faint opacity-0 group-hover/msg:opacity-100 hover:text-text-muted transition-[opacity,color] cursor-pointer"
              >
                {revisionLabel}
              </button>
            )}
            <AnimatePresence>
              {isReviseTarget && (
                <motion.p
                  key="revise-label"
                  initial={{ opacity: 1 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="absolute top-0 -translate-y-full text-xs text-text-faint"
                >
                  {revisionLabel === "Edit revision"
                    ? "Editing revision"
                    : "Adding revision"}
                </motion.p>
              )}
            </AnimatePresence>
          </div>
        )}
        <div
          className={`relative text-text text-sm w-fit leading-relaxed whitespace-pre-wrap`}
        >
          {content}
          {isStreaming && (
            <span className="inline-block w-0.5 h-4 ml-0.5 bg-text-muted align-middle animate-pulse" />
          )}
        </div>
      </div>
      {reviseSuggestions.length > 0 && (
        <div className="flex flex-col gap-1">
          {reviseSuggestions.map((r, i) => (
            <div key={i} className="group/rev flex items-center gap-2 w-fit">
              <code className="text-xs text-text-muted font-mono bg-surface rounded px-2 py-1 whitespace-pre-wrap break-words w-fit">
                {r.revision}
              </code>
              {onEdit && (
                <button
                  onClick={() => onEdit(r)}
                  className="shrink-0 text-xs text-text-faint opacity-0 group-hover/rev:opacity-100 hover:text-text-muted transition-[opacity,color] cursor-pointer"
                >
                  Edit
                </button>
              )}
              {onDelete && (
                <button
                  onClick={() => onDelete(r)}
                  className="shrink-0 text-xs text-text-faint opacity-0 group-hover/rev:opacity-100 hover:text-error transition-[opacity,color] cursor-pointer"
                >
                  Delete
                </button>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
