import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { TrainPanel } from "../components/TrainPanel";

const noop = () => {};

function makeProps(overrides = {}) {
  return {
    phase: 1 as const,
    setPhase: noop,
    userReward: 0.0,
    setUserReward: noop,
    estimatedReward: null,
    status: "idle" as const,
    statusMessage: "",
    history: [],
    onTrain: noop,
    onSave: noop,
    ...overrides,
  };
}

test("renders phase radio buttons", () => {
  render(<TrainPanel {...makeProps()} />);
  expect(screen.getByLabelText(/phase 1/i)).toBeInTheDocument();
  expect(screen.getByLabelText(/phase 2/i)).toBeInTheDocument();
});

test("shows reward slider in phase 1", () => {
  render(<TrainPanel {...makeProps({ phase: 1 })} />);
  expect(screen.getByText(/your reward/i)).toBeInTheDocument();
});

test("hides reward slider in phase 2", () => {
  render(<TrainPanel {...makeProps({ phase: 2 })} />);
  expect(screen.queryByText(/your reward/i)).not.toBeInTheDocument();
});

test("train button is disabled while training", () => {
  render(<TrainPanel {...makeProps({ status: "training" })} />);
  expect(screen.getByRole("button", { name: /training/i })).toBeDisabled();
});

test("shows estimated reward", () => {
  render(<TrainPanel {...makeProps({ estimatedReward: 0.77 })} />);
  expect(screen.getByText(/0\.77/)).toBeInTheDocument();
});

test("calls onTrain on button click", async () => {
  const onTrain = vi.fn();
  render(<TrainPanel {...makeProps({ onTrain })} />);
  await userEvent.click(screen.getByRole("button", { name: /^train$/i }));
  expect(onTrain).toHaveBeenCalled();
});
