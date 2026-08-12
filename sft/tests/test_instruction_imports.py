from pathlib import Path


def test_claude_instruction_files_import_their_full_agents_sibling():
    root = Path(__file__).resolve().parents[2]
    for directory in (root, root / "frontend", root / "models", root / "sft"):
        agents = directory / "AGENTS.md"
        claude = directory / "CLAUDE.md"
        assert agents.exists()
        assert claude.read_text() == "@AGENTS.md\n"
