import { useHealth } from "~/lib/useHealth";

export default function HealthBadge() {
  const { online, modelLoaded } = useHealth();

  return (
    <div className="flex items-center gap-1.5 text-sm">
      <span
        className={`size-2 rounded-full ${online ? "bg-ok" : "bg-error"}`}
      />
      <span className="text-text-muted">
        {online ? (modelLoaded ? "Model ready" : "Loading model…") : "Offline"}
      </span>
    </div>
  );
}
