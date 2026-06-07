import { useEffect, useState } from "react";
import { getHealth, type HealthResponse } from "~/lib/api";

export default function HealthBadge() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState(false);

  useEffect(() => {
    async function check() {
      try {
        const h = await getHealth();
        setHealth(h);
        setError(false);
      } catch {
        setHealth(null);
        setError(true);
      }
    }

    check();
    const interval = setInterval(check, 5000);
    return () => clearInterval(interval);
  }, []);

  const online = !error && health?.status === "ok";
  const modelLoaded = health?.model_loaded ?? false;

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
