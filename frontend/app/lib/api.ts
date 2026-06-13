const BACKEND_URL =
  (import.meta.env.VITE_BACKEND_URL as string | undefined) ??
  "http://localhost:8000";

export interface HealthResponse {
  status: string;
  model_loaded: boolean;
}

export interface ConfigResponse {
  user_open: string;
  asst_open: string;
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

export interface ReviseEntry {
  atTurn: number; // message index of the latest model response when this was recorded
  revision: string; // full "<revise back=N>...</revise weight=0.5>" tag string
}

export interface SessionData {
  messages: Message[];
  reviseSuggestions: ReviseEntry[];
}

export async function getHealth(): Promise<HealthResponse> {
  const res = await fetch(`${BACKEND_URL}/health`);
  if (!res.ok) throw new Error(`Health check failed: ${res.status}`);
  return res.json() as Promise<HealthResponse>;
}

export async function getConfig(): Promise<ConfigResponse> {
  const res = await fetch(`${BACKEND_URL}/config`);
  if (!res.ok) throw new Error(`Config fetch failed: ${res.status}`);
  return res.json() as Promise<ConfigResponse>;
}

export async function getSession(): Promise<SessionData> {
  const res = await fetch(`${BACKEND_URL}/session`);
  if (!res.ok) throw new Error(`Get session failed: ${res.status}`);
  const data = (await res.json()) as {
    messages: Message[];
    revise_suggestions: Array<{ at_turn: number; revision: string }>;
  };
  return {
    messages: data.messages,
    reviseSuggestions: data.revise_suggestions.map((e) => ({
      atTurn: e.at_turn,
      revision: e.revision,
    })),
  };
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

export async function submitRevision(
  n: number,
  revision: string,
  weight: number = 0.5,
  atTurn?: number,
): Promise<void> {
  const res = await fetch(`${BACKEND_URL}/session/revise`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      n,
      revision,
      weight,
      ...(atTurn !== undefined ? { at_turn: atTurn } : {}),
    }),
  });
  if (!res.ok) throw new Error(`Submit revision failed: ${res.status}`);
}

export async function deleteRevision(atTurn: number, n: number): Promise<void> {
  const res = await fetch(`${BACKEND_URL}/session/revise`, {
    method: "DELETE",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ at_turn: atTurn, n }),
  });
  if (!res.ok) throw new Error(`Delete revision failed: ${res.status}`);
}

export async function* streamGenerate(
  asstOpen: string,
  params?: {
    max_tokens?: number;
    temperature?: number;
    top_p?: number;
  },
): AsyncGenerator<GeneratedToken> {
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
  let prefixBuf = ""; // accumulates text until asstOpen is fully consumed

  function* processToken(token: GeneratedToken): Generator<GeneratedToken> {
    if (token.is_eos || token.token_id === 0) return;

    if (prefixBuf.length < asstOpen.length) {
      prefixBuf += token.token;
      if (prefixBuf.length >= asstOpen.length) {
        const remainder = prefixBuf.startsWith(asstOpen)
          ? prefixBuf.slice(asstOpen.length)
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
