import ulid
from typing import Literal

AttachmentPrefix = Literal[
    "file",
    "seg",
    "mask",
    "img",
    "report",
    "meta"
]


def new_id(prefix: AttachmentPrefix) -> str:
    """
    Generate an LLM-friendly, globally unique, type-prefixed ULID.

    Example:
        seg-01J5Z8M2A4YQY6H1P9T4KQZ8D2
    """
    return f"{prefix}-{str(ulid.new())}"
