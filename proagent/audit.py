"""Audit logging.

Every decision can be written out as one JSON line, which is what you'd want in
production: a replayable record of what the agent saw, which floor bound it, and
why. Off by default so tests and the eval stay quiet; turn it on by passing a
path to the agent or calling configure_audit.
"""
from __future__ import annotations

import json
import logging
from typing import Optional

logger = logging.getLogger("proagent.audit")
logger.addHandler(logging.NullHandler())


def configure_audit(path: Optional[str] = None, level: int = logging.INFO) -> None:
    """Send audit records to a file (JSON lines) or, if no path, to stderr."""
    logger.setLevel(level)
    handler: logging.Handler = logging.FileHandler(path) if path else logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def record(event: dict) -> None:
    if logger.isEnabledFor(logging.INFO):
        logger.info(json.dumps(event, sort_keys=True))
