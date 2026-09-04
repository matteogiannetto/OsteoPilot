from __future__ import annotations

import os
from datetime import datetime
from functools import lru_cache
from zoneinfo import ZoneInfo

DEFAULT_TIMEZONE = "Europe/Rome"


@lru_cache(maxsize=1)
def local_timezone() -> ZoneInfo:
    name = (
        os.environ.get("BIOMED_TIMEZONE")
        or os.environ.get("TZ")
        or DEFAULT_TIMEZONE
    )
    try:
        return ZoneInfo(name)
    except Exception:
        return ZoneInfo(DEFAULT_TIMEZONE)


def local_now() -> datetime:
    return datetime.now(local_timezone())


def local_now_iso() -> str:
    return local_now().isoformat()
