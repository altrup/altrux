import { postJSON } from "./client";
import type { GenerateResponse } from "../types";

export function generateFull(prompt: string, maxNewTokens: number): Promise<GenerateResponse> {
  return postJSON<GenerateResponse>("/generate", { prompt, max_new_tokens: maxNewTokens });
}
