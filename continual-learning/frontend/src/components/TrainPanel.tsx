import type { HistoryEntry, TrainStatus } from "../types";
import { HistoryTable } from "./HistoryTable";

interface Props {
  phase: 1 | 2;
  setPhase: (p: 1 | 2) => void;
  userReward: number;
  setUserReward: (r: number) => void;
  estimatedReward: number | null;
  status: TrainStatus;
  statusMessage: string;
  history: HistoryEntry[];
  onTrain: () => void;
  onSave: () => void;
}

export function TrainPanel({
  phase,
  setPhase,
  userReward,
  setUserReward,
  estimatedReward,
  status,
  statusMessage,
  history,
  onTrain,
  onSave,
}: Props) {
  const isBusy = status === "training";

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        gap: "0.75rem",
        width: "320px",
        flexShrink: 0,
        overflowY: "auto",
        padding: "1rem",
      }}
    >
      <h2 style={{ margin: 0 }}>Train</h2>

      <fieldset style={{ border: "1px solid #ddd", borderRadius: "4px", padding: "0.5rem" }}>
        <legend>Phase</legend>
        {([1, 2] as const).map((p) => (
          <label key={p} style={{ marginRight: "1rem" }}>
            <input
              type="radio"
              name="phase"
              value={p}
              checked={phase === p}
              onChange={() => setPhase(p)}
            />{" "}
            Phase {p}
          </label>
        ))}
      </fieldset>

      <div style={{ fontSize: "0.85rem", color: "#555" }}>
        Critic&apos;s estimated reward:{" "}
        <strong>{estimatedReward !== null ? estimatedReward.toFixed(3) : "—"}</strong>
      </div>

      {phase === 1 && (
        <label style={{ display: "flex", flexDirection: "column", gap: "0.25rem" }}>
          <span>Your reward: {userReward.toFixed(2)}</span>
          <input
            type="range"
            min={-1}
            max={1}
            step={0.01}
            value={userReward}
            onChange={(e) => setUserReward(Number(e.target.value))}
          />
        </label>
      )}

      <div style={{ display: "flex", gap: "0.5rem" }}>
        <button onClick={onTrain} disabled={isBusy}>
          {isBusy ? "Training…" : "Train"}
        </button>
        <button onClick={onSave} disabled={isBusy} style={{ background: "#f0f8ff" }}>
          Save
        </button>
      </div>

      <div style={{ fontSize: "0.82rem", color: "#666", minHeight: "1.2em" }}>{statusMessage}</div>

      <div>
        <h3 style={{ margin: "0 0 0.5rem" }}>Recent history</h3>
        <HistoryTable entries={history} />
      </div>
    </div>
  );
}
