import { useTheme } from "next-themes";
import { useEffect, useState } from "react";
import { LuMonitor, LuMoon, LuSun } from "react-icons/lu";

const themes = ["system", "light", "dark"] as const;
type Theme = (typeof themes)[number];

const labels: Record<Theme, string> = {
  system: "System",
  light: "Light",
  dark: "Dark",
};

const icons: Record<Theme, React.ReactNode> = {
  system: <LuMonitor size={16} />,
  light: <LuSun size={16} />,
  dark: <LuMoon size={16} />,
};

export default function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  const [mounted, setMounted] = useState(false);

  useEffect(() => setMounted(true), []);

  if (!mounted) return <div className="size-8" />;

  const current = (theme as Theme | undefined) ?? "system";

  function cycle() {
    const idx = themes.indexOf(current);
    setTheme(themes[(idx + 1) % themes.length]);
  }

  return (
    <button
      onClick={cycle}
      title={`Theme: ${labels[current]}`}
      aria-label={`Switch theme (current: ${labels[current]})`}
      className="size-8 rounded-lg flex items-center justify-center text-text-muted hover:text-text hover:bg-surface-raised transition-colors cursor-pointer"
    >
      {icons[current]}
    </button>
  );
}
