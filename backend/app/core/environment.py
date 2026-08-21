"""Repository-wide environment loading.

All entry points use the same root ``.env`` file, independently of the
process working directory.  Process variables keep precedence so deployment
systems can still override local defaults.
"""
from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ENV_PATH = PROJECT_ROOT / ".env"


def load_project_env() -> bool:
    """Load the single repository environment file."""

    return load_dotenv(PROJECT_ENV_PATH, override=False)


__all__ = ["PROJECT_ENV_PATH", "PROJECT_ROOT", "load_project_env"]
