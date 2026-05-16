import { useRef, useState } from "react";
import { streamGenerate } from "../api/client";
import type { GenerateStatus } from "../types";

export function useGenerate() {
  const [response, setResponse] = useState("");
  const [reward, setReward] = useState<number | null>(null);
  const [status, setStatus] = useState<GenerateStatus>("idle");
  const [prefillProgress, setPrefillProgress] = useState(0); // 0–1
  const [promptTokens, setPromptTokens] = useState<string[]>([]);
  const abortRef = useRef<(() => void) | null>(null);

  const generate = (prompt: string, maxNewTokens: number) => {
    abortRef.current?.();
    setResponse("");
    setReward(null);
    setPrefillProgress(0);
    setPromptTokens([]);
    setStatus("idle");

    abortRef.current = streamGenerate(
      prompt,
      maxNewTokens,
      (tokens) => { setPromptTokens(tokens); setStatus("prefill"); },
      (progress) => setPrefillProgress(progress),
      (text) => { setStatus("streaming"); setResponse(text); },
      (text, r) => { setResponse(text); setReward(r); setStatus("done"); },
      () => setStatus("error"),
    );
  };

  const abort = () => {
    abortRef.current?.();
    setStatus("idle");
    setPrefillProgress(0);
  };

  return { response, reward, status, prefillProgress, promptTokens, generate, abort };
}
