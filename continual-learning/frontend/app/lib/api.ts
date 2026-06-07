const BACKEND_URL =
  (import.meta.env.VITE_BACKEND_URL as string | undefined) ??
  "http://localhost:8000";

const USER_OPEN =
  (import.meta.env.VITE_USER_OPEN as string | undefined) ?? "[USER] ";

const ASST_OPEN =
  (import.meta.env.VITE_ASST_OPEN as string | undefined) ?? "[ASSISTANT] ";

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
}

export interface GeneratedToken {
  token: string;
  token_id: number;
  is_eos: boolean;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BACKEND_URL}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json() as Promise<HealthResponse>;
}

export async function getSession(): Promise<Message[]> {
  const res = await fetch(`${BACKEND_URL}/session`);
  if (!res.ok) throw new Error(`Get session failed: ${res.status}`);
  const { tokens } = (await res.json()) as {
    tokens: { id: number; text: string }[];
  };
  const text = tokens
    .filter((t) => t.id !== 0)
    .map((t) => t.text)
    .join("");
  return parseSession(text);
}

function parseSession(text: string): Message[] {
  const escape = (s: string) => s.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const regex = new RegExp(`(${escape(USER_OPEN)}|${escape(ASST_OPEN)})`, "g");

  const messages: Message[] = [];
  let currentRole: "user" | "assistant" | null = null;

  for (const part of text.split(regex)) {
    if (part === USER_OPEN) {
      currentRole = "user";
    } else if (part === ASST_OPEN) {
      currentRole = "assistant";
    } else if (currentRole) {
      const content = part.trim();
      if (content) messages.push({ role: currentRole, content });
    }
  }

  return messages;
}

export async function resetSession(): Promise<void> {
  const res = await fetch(`${BACKEND_URL}/session`, { method: "DELETE" });
  if (!res.ok) throw new Error(`Session reset failed: ${res.status}`);
}

export async function addUserMessage(content: string): Promise<void> {
  const res = await fetch(`${BACKEND_URL}/session/message`, {
    method: "PUT",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ role: "user", content }),
  });
  if (!res.ok) throw new Error(`Add message failed: ${res.status}`);
}

export async function* streamGenerate(params?: {
  max_tokens?: number;
  temperature?: number;
  top_p?: number;
}): AsyncGenerator<GeneratedToken> {
  const res = await fetch(`${BACKEND_URL}/generate/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      max_tokens: params?.max_tokens ?? 512,
      temperature: params?.temperature ?? 0.8,
      top_p: params?.top_p ?? 0.95,
    }),
  });

  if (!res.ok) throw new Error(`Generate failed: ${res.status}`);
  if (!res.body) throw new Error("No response body");

  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let prefixBuf = ""; // accumulates text until ASST_OPEN is fully consumed

  function* processToken(token: GeneratedToken): Generator<GeneratedToken> {
    if (token.is_eos || token.token_id === 0) return;

    if (prefixBuf.length < ASST_OPEN.length) {
      prefixBuf += token.token;
      if (prefixBuf.length >= ASST_OPEN.length) {
        const remainder = prefixBuf.startsWith(ASST_OPEN)
          ? prefixBuf.slice(ASST_OPEN.length)
          : prefixBuf;
        if (remainder) yield { ...token, token: remainder };
      }
      return;
    }

    yield token;
  }

  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop() ?? "";

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed) continue;
        yield* processToken(JSON.parse(trimmed) as GeneratedToken);
      }
    }

    if (buffer.trim()) {
      yield* processToken(JSON.parse(buffer.trim()) as GeneratedToken);
    }
  } finally {
    reader.releaseLock();
  }
}
