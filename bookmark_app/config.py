"""Configuration, environment loading, and logging setup."""

import logging
import os
import platform
from pathlib import Path

from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (overridable via environment variables)
# ---------------------------------------------------------------------------
LLM_MODEL = "gpt-4.1"
EMBEDDING_MODEL = "text-embedding-3-large"
VECTOR_STORE_DIR = "vector_store"
BOOKMARKS_CACHE_PATH = "all_bookmarks.json"
RETRIEVAL_K = 10
LOG_LEVEL = "INFO"


def load_env() -> None:
    """Load variables from a ``.env`` file if present."""
    load_dotenv()

    # Re-read tunables from env so that .env values take effect.
    global LLM_MODEL, EMBEDDING_MODEL, VECTOR_STORE_DIR, BOOKMARKS_CACHE_PATH
    global RETRIEVAL_K, LOG_LEVEL

    LLM_MODEL = os.getenv("LLM_MODEL", LLM_MODEL)
    EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL)
    VECTOR_STORE_DIR = os.getenv("VECTOR_STORE_DIR", VECTOR_STORE_DIR)
    BOOKMARKS_CACHE_PATH = os.getenv("BOOKMARKS_CACHE_PATH", BOOKMARKS_CACHE_PATH)
    RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", str(RETRIEVAL_K)))
    LOG_LEVEL = os.getenv("LOG_LEVEL", LOG_LEVEL)


def setup_logging() -> None:
    """Configure root logger with the level from ``LOG_LEVEL``."""
    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def get_bookmarks_path() -> Path:
    """Locate the Chrome bookmarks file.

    Uses the ``BOOKMARKS_PATH`` environment variable if set, otherwise
    auto-detects by OS.
    """
    env_path = os.environ.get("BOOKMARKS_PATH")
    if env_path:
        return Path(env_path).expanduser()

    system = platform.system()
    if system == "Windows":
        base = Path(os.environ.get("USERPROFILE", ""))
        return (
            base / "AppData" / "Local" / "Google" / "Chrome"
            / "User Data" / "Default" / "Bookmarks"
        )
    if system == "Darwin":
        return (
            Path.home() / "Library" / "Application Support"
            / "Google" / "Chrome" / "Default" / "Bookmarks"
        )
    # Linux / other Unix
    return Path.home() / ".config" / "google-chrome" / "Default" / "Bookmarks"


def validate_config() -> None:
    """Fail fast if required configuration is missing."""
    if not os.environ.get("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Add it to your .env file or export it."
        )

    bookmarks_path = get_bookmarks_path()
    if not bookmarks_path.exists():
        raise FileNotFoundError(
            f"Chrome bookmarks file not found at {bookmarks_path}. "
            "Set BOOKMARKS_PATH to the correct location."
        )
