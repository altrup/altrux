export interface GenerateResponse {
  response: string;
  estimated_reward: number;
}

export interface CriticTrainResponse {
  loss: number;
  step: number;
}

export interface PolicyTrainResponse {
  reward: number;
  loss: number;
  step: number;
}

export interface CheckpointStatusResponse {
  step: number;
  checkpoint_dir: string;
  last_saved_step: number | null;
}

export interface HistoryEntry {
  step: number;
  phase: 1 | 2;
  user_reward: number | null;
  predicted_reward: number;
  loss: number;
}

export interface HistoryResponse {
  entries: HistoryEntry[];
}

export type GenerateStatus = "idle" | "prefill" | "streaming" | "done" | "error";
export type TrainStatus = "idle" | "training" | "done" | "error";
