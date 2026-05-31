# TODO: Train the critic on collected data.
#
# Input files (data/collected/):
#   trunk_hiddens.dat  — raw float32 bytes; load with:
#                        np.fromfile(path, dtype=np.float32).reshape(-1, d_model)
#   records.jsonl      — {"index": N, "token_id": int, "reward": float} per line
#
# Training:
#   1. Load both files, pair by line index.
#   2. Forward trunk_hidden through critic_layers + critic_head → predicted_reward.
#   3. MSE or Huber loss vs stored reward.
#      Upweight nonzero rewards to avoid predicting-zero collapse.
#   4. Backprop into critic_layers + critic_head only (trunk frozen).
#   5. Checkpoint to checkpoints/critic_checkpoint.pt.
#
# Run as an offline script, not as a FastAPI endpoint.
