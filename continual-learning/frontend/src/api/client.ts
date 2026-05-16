export async function postJSON<T>(path: string, body: unknown): Promise<T> {
  const resp = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!resp.ok) {
    const text = await resp.text().catch(() => "");
    throw new Error(`${resp.status} ${resp.statusText}: ${text}`);
  }
  return resp.json() as Promise<T>;
}

export async function getJSON<T>(path: string, params?: Record<string, string | number>): Promise<T> {
  const url = params
    ? `${path}?${new URLSearchParams(Object.entries(params).map(([k, v]) => [k, String(v)]))}`
    : path;
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`${resp.status} ${resp.statusText}`);
  return resp.json() as Promise<T>;
}

export function streamGenerate(
  prompt: string,
  maxNewTokens: number,
  onPrefillStart: (tokens: string[]) => void,
  onPrefillProgress: (progress: number) => void,   // 0–1 from layer events
  onChunk: (text: string) => void,
  onDone: (text: string, reward: number) => void,
  onError: (e: Error) => void,
): () => void {
  const params = new URLSearchParams({
    prompt,
    max_new_tokens: String(maxNewTokens),
  });
  const es = new EventSource(`/generate/stream?${params}`);

  es.onmessage = (e: MessageEvent) => {
    if (e.data === "[DONE]") {
      es.close();
      return;
    }
    try {
      const parsed = JSON.parse(e.data as string) as {
        status?: string;
        prefill_layer?: number;
        total_layers?: number;
        chunk?: string;
        estimated_reward?: number;
      };

      if (parsed.status === "prefill") {
        onPrefillStart((parsed.tokens as string[]) ?? []);
      } else if (parsed.prefill_layer !== undefined && parsed.total_layers !== undefined) {
        onPrefillProgress(parsed.prefill_layer / parsed.total_layers);
      } else if (parsed.chunk !== undefined) {
        if (parsed.estimated_reward !== undefined) {
          onDone(parsed.chunk, parsed.estimated_reward);
        } else {
          onChunk(parsed.chunk);
        }
      }
    } catch {
      // ignore malformed lines
    }
  };

  es.onerror = () => {
    es.close();
    onError(new Error("SSE connection error"));
  };

  return () => es.close();
}
