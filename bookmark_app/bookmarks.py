"""Chrome bookmark extraction and JSON cache management."""

import json
import logging
import pickle
from pathlib import Path

from .config import get_bookmarks_path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chrome extraction
# ---------------------------------------------------------------------------

def extract_bookmarks(nodes: list, parent_folder: str = "") -> list[dict]:
    """Recursively flatten Chrome bookmark nodes into a list of dicts."""
    extracted: list[dict] = []
    for item in nodes:
        if "children" in item:
            folder_path = f"{parent_folder}/{item['name']}"
            extracted.extend(extract_bookmarks(item["children"], folder_path))
        elif "url" in item:
            extracted.append({
                "folder": parent_folder,
                "name": item["name"],
                "url": item["url"],
            })
    return extracted


def load_chrome_bookmarks(path: Path | None = None) -> list[dict]:
    """Read the Chrome Bookmarks JSON and return a flat bookmark list."""
    path = path or get_bookmarks_path()
    logger.info("Reading Chrome bookmarks from %s", path)

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    all_bookmarks: list[dict] = []
    for root in ("bookmark_bar", "other", "synced"):
        if root in data["roots"]:
            all_bookmarks.extend(
                extract_bookmarks(data["roots"][root]["children"])
            )

    logger.info("Extracted %d bookmarks from Chrome", len(all_bookmarks))
    return all_bookmarks


# ---------------------------------------------------------------------------
# Cache (JSON with legacy pickle migration)
# ---------------------------------------------------------------------------

def load_cache(path: str) -> list[dict]:
    """Load the bookmark cache from *path* (JSON).

    If *path* does not exist but a legacy ``all_bookmarks.pkl`` does, migrate
    it to JSON automatically.
    """
    json_path = Path(path)

    if json_path.exists():
        logger.info("Loading bookmark cache from %s", json_path)
        with json_path.open("r", encoding="utf-8") as f:
            return json.load(f)

    # Legacy pickle migration
    pkl_path = Path("all_bookmarks.pkl")
    if pkl_path.exists():
        logger.info("Migrating legacy pickle cache → %s", json_path)
        with pkl_path.open("rb") as f:
            data = pickle.load(f)  # noqa: S301
        save_cache(data, path)
        return data

    logger.info("No existing bookmark cache found; starting fresh")
    return []


def save_cache(bookmarks: list[dict], path: str) -> None:
    """Persist bookmarks to a JSON file."""
    json_path = Path(path)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(bookmarks, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d bookmarks to %s", len(bookmarks), json_path)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_bookmarks(fresh: list[dict], cached: list[dict]) -> list[dict]:
    """Merge freshly extracted bookmarks with the cached list.

    Cached bookmarks that still exist in *fresh* keep their descriptions.
    New bookmarks (present in *fresh* but not in *cached*) are appended
    without a description so the description generator picks them up.
    Bookmarks removed from Chrome (in *cached* but not in *fresh*) are dropped.
    """
    cached_by_url: dict[str, dict] = {b["url"]: b for b in cached}
    merged: list[dict] = []

    for bookmark in fresh:
        url = bookmark["url"]
        if url in cached_by_url:
            # Preserve existing entry (keeps description if present)
            merged.append(cached_by_url[url])
        else:
            merged.append(bookmark)

    new_count = sum(1 for b in merged if "description" not in b)
    logger.info(
        "Merged bookmarks: %d total, %d new (need descriptions)",
        len(merged), new_count,
    )
    return merged
