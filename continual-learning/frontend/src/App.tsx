import { useRef } from "react";
import { GeneratePanel } from "./components/GeneratePanel";
import { TrainPanel } from "./components/TrainPanel";
import { useGenerate } from "./hooks/useGenerate";
import { useTrain } from "./hooks/useTrain";

export function App() {
  const gen = useGenerate();
  const tr = useTrain();
  const promptRef = useRef("");

  const handleGenerate = (prompt: string, maxNewTokens: number) => {
    promptRef.current = prompt;
    gen.generate(prompt, maxNewTokens);
  };

  const handleTrain = () => {
    tr.train(promptRef.current, gen.response);
  };

  return (
    <div
      style={{
        display: "flex",
        height: "100vh",
        fontFamily: "system-ui, sans-serif",
        overflow: "hidden",
      }}
    >
      <GeneratePanel
        response={gen.response}
        reward={gen.reward}
        status={gen.status}
        prefillProgress={gen.prefillProgress}
        promptTokens={gen.promptTokens}
        onGenerate={handleGenerate}
        onAbort={gen.abort}
      />
      <div style={{ width: "1px", background: "#e0e0e0", flexShrink: 0 }} />
      <TrainPanel
        phase={tr.phase}
        setPhase={tr.setPhase}
        userReward={tr.userReward}
        setUserReward={tr.setUserReward}
        estimatedReward={gen.reward}
        status={tr.status}
        statusMessage={tr.statusMessage}
        history={tr.history}
        onTrain={handleTrain}
        onSave={tr.save}
      />
    </div>
  );
}
