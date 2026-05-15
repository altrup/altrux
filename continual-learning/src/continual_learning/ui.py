import gradio as gr
from .model import ContinualLearningModel
from .trainer import Trainer


PHASE_LABELS = {
    1: "Phase 1 — Train critic",
    2: "Phase 2 — Train base model",
}


def create_ui(model: ContinualLearningModel, trainer: Trainer, initial_phase: int = 1) -> gr.Blocks:
    session: dict = {}

    def on_generate(prompt: str, max_tokens: int):
        if not prompt.strip():
            yield "", 0.0, "Enter a prompt first."
            return

        for text, reward in model.generate_stream(prompt, max_new_tokens=int(max_tokens)):
            if reward is None:
                yield text, 0.0, "Generating…"
            else:
                session["prompt"] = prompt
                session["response"] = text
                yield text, round(reward, 3), f"Done. Critic's estimated reward: {reward:.3f}"

    def on_save():
        trainer.save_now()
        return f"Saved to {trainer.config.checkpoint_dir}/ (step {len(trainer.history)})"

    def on_train(phase: str, user_reward: float):
        if "prompt" not in session:
            return "Generate a response first.", []

        if phase == "Phase 1 — Train critic":
            loss = trainer.critic_step(session["prompt"], session["response"], float(user_reward))
            status = f"Critic step complete. Loss: {loss:.4f}"
        else:
            reward_val, loss = trainer.policy_step(session["prompt"], session["response"])
            status = f"Policy step complete. Critic reward: {reward_val:.3f}  Loss: {loss:.4f}"

        rows = [
            [
                str(len(trainer.history) - len(trainer.history[-10:]) + i + 1),
                str(h.get("phase", "?")),
                f"{h.get('user_reward', '—') if isinstance(h.get('user_reward'), float) else '—'}",
                f"{h['predicted_reward']:.3f}",
                f"{h['loss']:.4f}",
            ]
            for i, h in enumerate(trainer.history[-10:])
        ]

        return status, rows

    with gr.Blocks(title="Continual Learning") as ui:
        gr.Markdown(
            "# Continual Learning\n"
            "**Phase 1:** rate responses to train the critic.  \n"
            "**Phase 2:** let the trained critic guide the base model."
        )

        with gr.Row():
            with gr.Column(scale=2):
                prompt_box = gr.Textbox(
                    label="Prompt", lines=4, placeholder="Enter your prompt…"
                )
                max_tokens = gr.Slider(16, 512, value=128, step=16, label="Max New Tokens")
                gen_btn = gr.Button("Generate", variant="primary")

                response_box = gr.Textbox(label="Response", lines=8, interactive=False)
                status_box = gr.Textbox(label="Status", interactive=False, lines=1)

            with gr.Column(scale=1):
                phase_radio = gr.Radio(
                    choices=list(PHASE_LABELS.values()),
                    value=PHASE_LABELS[initial_phase],
                    label="Training Phase",
                )
                estimated_box = gr.Number(
                    label="Critic's Estimated Reward", interactive=False, value=0.0
                )
                reward_slider = gr.Slider(
                    -1.0, 1.0, value=0.0, step=0.05, label="Your Reward Signal (Phase 1 only)"
                )
                train_btn = gr.Button("Train", variant="secondary")
                save_btn = gr.Button("Save Checkpoint", variant="secondary")
                train_status = gr.Textbox(label="Training Status", interactive=False, lines=1)

                gr.Markdown("### History (last 10)")
                history_table = gr.Dataframe(
                    headers=["Step", "Phase", "User Reward", "Predicted", "Loss"],
                    datatype=["str", "str", "str", "str", "str"],
                    interactive=False,
                )

        gen_btn.click(
            fn=on_generate,
            inputs=[prompt_box, max_tokens],
            outputs=[response_box, estimated_box, status_box],
        )

        train_btn.click(
            fn=on_train,
            inputs=[phase_radio, reward_slider],
            outputs=[train_status, history_table],
        )

        save_btn.click(
            fn=on_save,
            inputs=[],
            outputs=[train_status],
        )

    return ui
