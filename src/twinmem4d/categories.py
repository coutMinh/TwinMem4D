"""Map object names and DAM descriptions to a small category vocabulary."""

import re
from typing import Optional, Sequence

DEFAULT_VOCAB = ("bowl", "plate", "mug", "cup", "basket", "tray", "box", "bottle", "gripper", "robot", "table")


def category_of(text: str, vocab: Sequence[str] = DEFAULT_VOCAB) -> Optional[str]:
    """Vocabulary word that appears first in ``text`` (``a plate with a bowl`` -> plate), else None."""
    text = text.lower().replace("_", " ")
    best, best_pos = None, len(text) + 1
    for word in vocab:
        m = re.search(rf"\b{re.escape(word)}s?\b", text)
        if m and m.start() < best_pos:
            best, best_pos = word, m.start()
    return best
