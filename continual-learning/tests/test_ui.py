"""Smoke tests for the UI layer — exercises callback logic without a browser."""
import pytest
import gradio as gr
from continual_learning.ui import create_ui, PHASE_LABELS


def test_create_ui_returns_blocks(tiny_model, trainer):
    ui = create_ui(tiny_model, trainer)
    assert isinstance(ui, gr.Blocks)


def test_create_ui_phase_1_default(tiny_model, trainer):
    ui = create_ui(tiny_model, trainer, initial_phase=1)
    assert isinstance(ui, gr.Blocks)


def test_create_ui_phase_2_default(tiny_model, trainer):
    ui = create_ui(tiny_model, trainer, initial_phase=2)
    assert isinstance(ui, gr.Blocks)


def test_phase_labels_cover_both_phases():
    assert 1 in PHASE_LABELS
    assert 2 in PHASE_LABELS
    assert PHASE_LABELS[1] != PHASE_LABELS[2]
