import { postJSON, getJSON } from "./client";
import type {
  CriticTrainResponse,
  PolicyTrainResponse,
  HistoryResponse,
} from "../types";

export function trainCritic(
  prompt: string,
  response: string,
  userReward: number,
): Promise<CriticTrainResponse> {
  return postJSON<CriticTrainResponse>("/train/critic", {
    prompt,
    response,
    user_reward: userReward,
  });
}

export function trainPolicy(
  prompt: string,
  response: string,
): Promise<PolicyTrainResponse> {
  return postJSON<PolicyTrainResponse>("/train/policy", { prompt, response });
}

export function getHistory(n = 10): Promise<HistoryResponse> {
  return getJSON<HistoryResponse>("/train/history", { n });
}

export function saveCheckpoint(): Promise<{ ok: boolean; step: number }> {
  return postJSON("/checkpoint/save", {});
}
