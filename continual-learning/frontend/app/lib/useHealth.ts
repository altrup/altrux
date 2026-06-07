import { useEffect, useState } from "react";
import { getHealth, type HealthResponse } from "./api";

export function useHealth() {
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
  return { online, modelLoaded };
}
