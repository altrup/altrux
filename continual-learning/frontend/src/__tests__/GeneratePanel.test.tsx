import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { GeneratePanel } from "../components/GeneratePanel";

const noop = () => {};

function makeProps(overrides = {}) {
  return {
    response: "",
    reward: null,
    status: "idle" as const,
    prefillProgress: 0,
    promptTokens: [],
    onGenerate: noop,
    onAbort: noop,
    ...overrides,
  };
}

test("renders input textarea and send button", () => {
  render(<GeneratePanel {...makeProps()} />);
  expect(screen.getByPlaceholderText(/type a message/i)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: /send/i })).toBeInTheDocument();
});

test("send button is disabled when input is empty", () => {
  render(<GeneratePanel {...makeProps()} />);
  expect(screen.getByRole("button", { name: /send/i })).toBeDisabled();
});

test("input is disabled while prefilling", () => {
  render(<GeneratePanel {...makeProps({ status: "prefill" })} />);
  expect(screen.getByPlaceholderText(/type a message/i)).toBeDisabled();
});

test("input is disabled while streaming", () => {
  render(<GeneratePanel {...makeProps({ status: "streaming" })} />);
  expect(screen.getByPlaceholderText(/type a message/i)).toBeDisabled();
});

test("stop button replaces send while busy", () => {
  render(<GeneratePanel {...makeProps({ status: "streaming" })} />);
  expect(screen.getByRole("button", { name: /stop/i })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: /send/i })).not.toBeInTheDocument();
});

test("stop button shown during prefill", () => {
  render(<GeneratePanel {...makeProps({ status: "prefill" })} />);
  expect(screen.getByRole("button", { name: /stop/i })).toBeInTheDocument();
});

test("shows reward when done", () => {
  render(<GeneratePanel {...makeProps({ status: "done", reward: 0.42 })} />);
  expect(screen.getByText(/0\.42/)).toBeInTheDocument();
});

test("calls onGenerate with input text and token count", async () => {
  const onGenerate = vi.fn();
  render(<GeneratePanel {...makeProps({ onGenerate })} />);
  await userEvent.type(screen.getByPlaceholderText(/type a message/i), "hello");
  await userEvent.click(screen.getByRole("button", { name: /send/i }));
  expect(onGenerate).toHaveBeenCalledWith("hello", expect.any(Number));
});

test("clears input after send", async () => {
  render(<GeneratePanel {...makeProps()} />);
  const input = screen.getByPlaceholderText(/type a message/i);
  await userEvent.type(input, "hello");
  await userEvent.click(screen.getByRole("button", { name: /send/i }));
  expect(input).toHaveValue("");
});
