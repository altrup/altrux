import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[2]))

import topic_choice
from diagnostics import topic_choice as implementation


def test_topic_choice_reexports_diagnostic_surface():
    assert topic_choice.OPENERS is implementation.OPENERS
    assert topic_choice.main is implementation.main
