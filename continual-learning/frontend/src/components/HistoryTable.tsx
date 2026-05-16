import type { HistoryEntry } from "../types";

interface Props {
  entries: HistoryEntry[];
}

export function HistoryTable({ entries }: Props) {
  if (entries.length === 0) {
    return <p style={{ color: "#888", fontSize: "0.85rem" }}>No training history yet.</p>;
  }

  return (
    <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "0.82rem" }}>
      <thead>
        <tr style={{ borderBottom: "1px solid #ddd", textAlign: "right" }}>
          <th style={{ textAlign: "left" }}>Step</th>
          <th>Phase</th>
          <th>User Reward</th>
          <th>Predicted</th>
          <th>Loss</th>
        </tr>
      </thead>
      <tbody>
        {entries.map((e) => (
          <tr key={e.step} style={{ borderBottom: "1px solid #f0f0f0", textAlign: "right" }}>
            <td style={{ textAlign: "left" }}>{e.step}</td>
            <td>{e.phase}</td>
            <td>{e.user_reward !== null ? e.user_reward.toFixed(3) : "—"}</td>
            <td>{e.predicted_reward.toFixed(3)}</td>
            <td>{e.loss.toFixed(4)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
