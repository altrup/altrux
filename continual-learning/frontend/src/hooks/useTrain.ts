import { useState } from "react";
import { trainCritic, trainPolicy, getHistory, saveCheckpoint } from "../api/train";
import type { HistoryEntry, TrainStatus } from "../types";

export function useTrain() {
  const [phase, setPhase] = useState<1 | 2>(1);
  const [userReward, setUserReward] = useState(0.0);
  const [status, setStatus] = useState<TrainStatus>("idle");
  const [statusMessage, setStatusMessage] = useState("");
  const [history, setHistory] = useState<HistoryEntry[]>([]);

  const refreshHistory = async () => {
    const data = await getHistory(10);
    setHistory(data.entries);
  };

  const train = async (prompt: string, response: string) => {
    setStatus("training");
    try {
      if (phase === 1) {
        const result = await trainCritic(prompt, response, userReward);
        setStatusMessage(`Critic step ${result.step} — loss: ${result.loss.toFixed(4)}`);
      } else {
        const result = await trainPolicy(prompt, response);
        setStatusMessage(
          `Policy step ${result.step} — reward: ${result.reward.toFixed(3)}, loss: ${result.loss.toFixed(4)}`,
        );
      }
      setStatus("done");
      await refreshHistory();
    } catch (e) {
      setStatusMessage(`Error: ${String(e)}`);
      setStatus("error");
    }
  };

  const save = async () => {
    setStatus("training");
    try {
      const result = await saveCheckpoint();
      setStatusMessage(`Saved at step ${result.step}.`);
      setStatus("done");
    } catch (e) {
      setStatusMessage(`Save failed: ${String(e)}`);
      setStatus("error");
    }
  };

  return {
    phase,
    setPhase,
    userReward,
    setUserReward,
    status,
    statusMessage,
    history,
    train,
    save,
    refreshHistory,
  };
}
