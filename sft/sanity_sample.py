"""Quick cross-slice sanity read before training: a few decoded windows and
headline counts from each data artifact -- NOT a substitute for the full
generation-time validation, just the ten-second read that catches a slice
whose contents don't match its name (root CLAUDE.md: no dataset goes to a
training run until someone has read a sample).

  make sanity-sample ARGS="--data data/train_chains.pt --data data/train_cram.pt"

Windows prefer recall-credited spans (cram/needles: shows a cue->answer with
its credit), then sleep positions (chains: shows a wipe boundary), then random
text. Output is meant to be handed to a reviewer (a cheap subagent on the
box) with the question: does each slice look like what it claims to be?
"""

from preparation.inspection import main, pick_windows

__all__ = ["pick_windows", "main"]


if __name__ == "__main__":
    main()
